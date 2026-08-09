package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	bigBangBigCrunchPopulation = 20
	bigBangBigCrunchMaxSteps   = 500
	bigBangBigCrunchTolerance  = 1e-6
	bigBangBigCrunchRange      = 15.0
	bigBangBigCrunchEpsilon    = 1e-10
)

// BigBangBigCrunch minimizuje sphere funkciju koristeći Big Bang - Big Crunch algoritam: svaka runda rasipa
// populaciju oko trenutnog centra mase sa smanjujućim radijusom (big bang),
// a zatim ponovo računa centar mase kao prosek ponderisan fitnesom (big crunch)
// problem.Point inicijalizuje centar mase i određuje dimenzionalnost
func BigBangBigCrunch(problem models.Problem) (result models.Result, err error) {

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

	population := bigBangBigCrunchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := bigBangBigCrunchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := bigBangBigCrunchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	searchRange := bigBangBigCrunchRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	centerOfMass := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), centerOfMass...)
	bestValue := fn.Evaluate(best)

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		radius := searchRange * (1 - float64(steps)/float64(maxSteps))

		points := make([][]float64, population)
		for i := range points {
			point := make([]float64, dimensions)
			for d := range point {
				point[d] = centerOfMass[d] + radius*randNorm()

				if point[d] > searchRange {
					point[d] = searchRange
				} else if point[d] < -searchRange {
					point[d] = -searchRange
				}
			}
			points[i] = point

			if value := fn.Evaluate(point); value < bestValue {
				bestValue = value
				best = append([]float64(nil), point...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		newCenter := make([]float64, dimensions)
		totalWeight := 0.0
		for _, point := range points {
			weight := 1 / (fn.Evaluate(point) + bigBangBigCrunchEpsilon)
			totalWeight += weight
			for d := range point {
				newCenter[d] += point[d] * weight
			}
		}
		for d := range newCenter {
			newCenter[d] /= totalWeight
		}

		centerOfMass = newCenter
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "big_bang_big_crunch",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
