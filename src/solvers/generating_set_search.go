package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	generatingSetSearchMaxSteps  = 1000
	generatingSetSearchTolerance = 1e-6
	generatingSetSearchStepSize  = 1.0
	generatingSetSearchStepClip  = 1.0
)

// GeneratingSetSearch minimizuje konfigurisanu benchmark funkciju koristeći Generating Set Search (GSS): pozitivni
// razapinjući skup pravaca [e1,-e1,e2,-e2,...] se testira U CELOSTI svaki korak (svi pravci pre odluke), za razliku
// od compass_search.go koji ide redom dok prvi uspešan pravac ne pomeri tačku — ovde se bira NAJBOLJI od svih
// pravaca koji poboljšavaju, step_size raste (×1.2) na uspeh i opada (×0.5) kad nijedan pravac ne pomogne
// problem.Point je početna tačka pretrage
func GeneratingSetSearch(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := generatingSetSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := generatingSetSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	stepSize := generatingSetSearchStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := append([]float64(nil), problem.Point...)

	directions := make([][]float64, 0, 2*dimensions)
	for d := 0; d < dimensions; d++ {
		pos := make([]float64, dimensions)
		pos[d] = 1.0
		neg := make([]float64, dimensions)
		neg[d] = -1.0
		directions = append(directions, pos, neg)
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		currentValue := fn.Evaluate(x)
		bestDirIdx := -1
		bestTrialValue := currentValue
		for i, d := range directions {
			step := make([]float64, dimensions)
			for k := range step {
				step[k] = stepSize * d[k]
			}
			step = clipStep(step, generatingSetSearchStepClip)
			trial := make([]float64, dimensions)
			for k := range trial {
				trial[k] = x[k] + step[k]
			}
			if v := fn.Evaluate(trial); v < bestTrialValue {
				bestTrialValue = v
				bestDirIdx = i
			}
		}

		if bestDirIdx >= 0 {
			step := make([]float64, dimensions)
			for k := range step {
				step[k] = stepSize * directions[bestDirIdx][k]
			}
			step = clipStep(step, generatingSetSearchStepClip)
			for k := range x {
				x[k] += step[k]
			}
			stepSize *= 1.2
		} else {
			stepSize *= 0.5
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "generating_set_search",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
