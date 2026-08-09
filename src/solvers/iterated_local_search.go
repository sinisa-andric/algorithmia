package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	iteratedLocalSearchPerturbationSize = 2.0
	iteratedLocalSearchLocalSteps       = 50
	iteratedLocalSearchMaxSteps         = 200
	iteratedLocalSearchTolerance        = 1e-6
	iteratedLocalSearchLocalLearnRate   = 0.1
)

// IteratedLocalSearch minimizuje sphere funkciju koristeći iterated local search: ponavljano izvršava lokalni
// gradijentni spust, a zatim perturbuje najbolje pronađeno rešenje kao polaznu tačku za sledeći krug
// problem.Point je početna tačka pretrage
func IteratedLocalSearch(problem models.Problem) (result models.Result, err error) {

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

	perturbationSize := iteratedLocalSearchPerturbationSize
	if v, ok := problem.Payload["perturbation_size"].(float64); ok {
		perturbationSize = v
	}

	localSteps := iteratedLocalSearchLocalSteps
	if v, ok := problem.Payload["local_steps"].(float64); ok {
		localSteps = int(v)
	}

	maxSteps := iteratedLocalSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := iteratedLocalSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	point := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), point...)
	bestValue := fn.Evaluate(best)

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for s := 0; s < localSteps; s++ {
			gradient := fn.Gradient(point)
			for d := range point {
				point[d] -= iteratedLocalSearchLocalLearnRate * gradient[d]
			}
		}

		if value := fn.Evaluate(point); value < bestValue {
			bestValue = value
			best = append([]float64(nil), point...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		point = make([]float64, dimensions)
		for d := range point {
			point[d] = best[d] + perturbationSize*randNorm()
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "iterated_local_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
