package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	scheduleFreeSgdLearningRate = 0.1
	scheduleFreeSgdMaxSteps     = 1000
	scheduleFreeSgdTolerance    = 1e-6
	scheduleFreeSgdStepClip     = 1.0
)

// ScheduleFreeSgd minimizuje konfigurisanu benchmark funkciju koristeći Schedule-Free SGD: isto kao
// schedule_free_adam.go, ali je z sopstvena istrajna sekvenca koja radi čist SGD korak (bez Adam adaptivnog
// koraka) — izveštena pozicija x je i dalje Polyak prosek (težina 1/(step+1)) te z sekvence
// NAPOMENA: ista ispravka kao u schedule_free_adam.go — z mora biti sopstvena istorija, ne preračunata iz
// trenutnog x svakog koraka (to bi implicitno uvelo 1/t opadanje stope učenja)
// problem.Point je početna tačka pretrage
func ScheduleFreeSgd(problem models.Problem) (result models.Result, err error) {

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

	learningRate := scheduleFreeSgdLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	maxSteps := scheduleFreeSgdMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := scheduleFreeSgdTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	z := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, z, proximalGradEps)
		t := float64(steps + 1)

		zDelta := make([]float64, dimensions)
		for d := range zDelta {
			zDelta[d] = -learningRate * grad[d]
		}
		zDelta = clipStep(zDelta, scheduleFreeSgdStepClip)
		for d := range z {
			z[d] += zDelta[d]
		}

		weight := 1 / t
		for d := range x {
			x[d] = (1-weight)*x[d] + weight*z[d]
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
		Method:     "schedule_free_sgd",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
