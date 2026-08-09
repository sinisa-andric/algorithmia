package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	activeSetBoundMaxSteps     = 1000
	activeSetBoundTolerance    = 1e-6
	activeSetBoundLearningRate = 0.1
	activeSetBoundStepClip     = 1.0
	activeSetBoundLower        = -5.0
	activeSetBoundUpper        = 5.0
)

// ActiveSetBound minimizuje konfigurisanu benchmark funkciju koristeći pojednostavljenu Active-Set metodu
// ograničenu na box aktivne granice (NE opšti QP active-set sa proizvoljnim linearnim ograničenjima): dimenzije
// "zaglavljene" na granici -5 ili +5 se isključuju iz gradijentnog koraka (active skup) dok gradijent ne počne
// da gura ka unutra, kada se granica oslobađa
// problem.Point je početna tačka pretrage (projektovana u [-5,5] ako je van opsega)
func ActiveSetBound(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := activeSetBoundMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := activeSetBoundTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	learningRate := activeSetBoundLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	lower, upper := activeSetBoundLower, activeSetBoundUpper

	x := make([]float64, dimensions)
	active := make([]bool, dimensions)
	for d := range x {
		x[d] = clamp(problem.Point[d], lower, upper)
		if x[d] == lower || x[d] == upper {
			active[d] = true
		}
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			if !active[d] {
				delta[d] = -learningRate * grad[d]
				continue
			}
			// dimenzija je na granici — oslobodi je iz active skupa samo ako gradijent gura ka unutra
			if x[d] == lower && grad[d] < 0 {
				active[d] = false
				delta[d] = -learningRate * grad[d]
			} else if x[d] == upper && grad[d] > 0 {
				active[d] = false
				delta[d] = -learningRate * grad[d]
			}
		}
		delta = clipStep(delta, activeSetBoundStepClip)

		for d := range x {
			x[d] += delta[d]
			if x[d] <= lower {
				x[d] = lower
				active[d] = true
			}
			if x[d] >= upper {
				x[d] = upper
				active[d] = true
			}
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "active_set_bound",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
