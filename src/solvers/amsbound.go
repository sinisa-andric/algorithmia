package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	amsboundLearningRate = 0.1
	amsboundBeta1        = 0.9
	amsboundBeta2        = 0.999
	amsboundEpsilon      = 1e-8
	amsboundFinalLr      = 0.1
	amsboundGamma        = 0.001
	amsboundMaxSteps     = 1000
	amsboundTolerance    = 1e-6
	amsboundStepClip     = 1.0
)

// Amsbound minimizuje konfigurisanu benchmark funkciju koristeći AMSBound: kombinuje AMSGrad tekući maksimum
// drugog momenta I dinamički bound na step_size (kao adabound.go) — razlika od adabound.go koji nema max-tracking,
// i od amsgrad.go koji nema dinamički bound
// problem.Point je početna tačka pretrage
func Amsbound(problem models.Problem) (result models.Result, err error) {

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

	learningRate := amsboundLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := amsboundBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := amsboundBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := amsboundEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	finalLr := amsboundFinalLr
	if v, ok := problem.Payload["final_lr"].(float64); ok {
		finalLr = v
	}

	gamma := amsboundGamma
	if v, ok := problem.Payload["gamma"].(float64); ok {
		gamma = v
	}

	maxSteps := amsboundMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := amsboundTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)
	sMax := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		t := float64(steps + 1)
		lower := finalLr * (1 - 1/(gamma*t+1))
		upper := finalLr * (1 + 1/(gamma*t))

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			sMax[d] = math.Max(sMax[d], s[d])

			stepSize := clamp(learningRate/(math.Sqrt(sMax[d])+epsilon), lower, upper)
			delta[d] = -stepSize * m[d]
		}
		delta = clipStep(delta, amsboundStepClip)
		for d := range x {
			x[d] += delta[d]
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
		Method:     "amsbound",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
