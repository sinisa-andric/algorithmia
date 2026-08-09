package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	basinHoppingStepSize       = 1.0
	basinHoppingTemperature    = 1.0
	basinHoppingMaxSteps       = 500
	basinHoppingLocalSteps     = 20
	basinHoppingTolerance      = 1e-6
	basinHoppingLocalLearnRate = 0.1
)

// BasinHopping minimizuje sphere funkciju koristeći basin hopping: nasumična perturbacija trenutnog rešenja praćena
// lokalnim gradijentnim spustom, pri čemu se novo rešenje prihvata preko Metropolis kriterijuma
// problem.Point je početna tačka pretrage
func BasinHopping(problem models.Problem) (result models.Result, err error) {

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

	stepSize := basinHoppingStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	temperature := basinHoppingTemperature
	if v, ok := problem.Payload["temperature"].(float64); ok {
		temperature = v
	}

	maxSteps := basinHoppingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	localSteps := basinHoppingLocalSteps
	if v, ok := problem.Payload["local_steps"].(float64); ok {
		localSteps = int(v)
	}

	tolerance := basinHoppingTolerance
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

		candidate := make([]float64, dimensions)
		for d := range candidate {
			candidate[d] = point[d] + stepSize*randNorm()
		}

		for s := 0; s < localSteps; s++ {
			gradient := fn.Gradient(candidate)
			for d := range candidate {
				candidate[d] -= basinHoppingLocalLearnRate * gradient[d]
			}
		}

		delta := fn.Evaluate(candidate) - fn.Evaluate(point)
		if delta < 0 || rand.Float64() < math.Exp(-delta/temperature) {
			point = candidate
		}

		if value := fn.Evaluate(point); value < bestValue {
			bestValue = value
			best = append([]float64(nil), point...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "basin_hopping",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
