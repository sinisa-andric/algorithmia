package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	gridSearchPointsPerDim = 20
	gridSearchLower        = -5.0
	gridSearchUpper        = 5.0
)

// GridSearch minimizuje konfigurisanu benchmark funkciju potpunom enumeracijom: generiše ravnomernu mrežu tačaka
// (grid_points_per_dim po dimenziji) preko celog opsega [-5,5]^n i vraća najbolju pronađenu — potpuno
// deterministička metoda bez ikakve iterativne pretrage; "steps" ovde predstavlja ukupan broj evaluiranih tačaka
// mreže, ne broj koraka optimizacije
// problem.Point određuje samo dimenziju pretrage, ne polaznu tačku (grid pokriva ceo opseg nezavisno od nje)
func GridSearch(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	pointsPerDim := gridSearchPointsPerDim
	if v, ok := problem.Payload["grid_points_per_dim"].(float64); ok {
		pointsPerDim = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	axisValues := make([]float64, pointsPerDim)
	for i := 0; i < pointsPerDim; i++ {
		if pointsPerDim == 1 {
			axisValues[i] = (gridSearchLower + gridSearchUpper) / 2
		} else {
			axisValues[i] = gridSearchLower + float64(i)*(gridSearchUpper-gridSearchLower)/float64(pointsPerDim-1)
		}
	}

	total := 1
	for d := 0; d < dimensions; d++ {
		total *= pointsPerDim
	}

	best := make([]float64, dimensions)
	for d := range best {
		best[d] = axisValues[0]
	}
	bestValue := fn.Evaluate(best)

	indices := make([]int, dimensions)
	steps := 0
	for steps = 0; steps < total; steps++ {
		point := make([]float64, dimensions)
		for d := range point {
			point[d] = axisValues[indices[d]]
		}
		if value := fn.Evaluate(point); value < bestValue {
			bestValue = value
			best = point
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		for d := dimensions - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < pointsPerDim {
				break
			}
			indices[d] = 0
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "grid_search",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
