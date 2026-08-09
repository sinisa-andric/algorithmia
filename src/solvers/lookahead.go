package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	lookaheadLearningRate = 0.001
	lookaheadK            = 5
	lookaheadAlpha        = 0.5
	lookaheadBeta1        = 0.9
	lookaheadBeta2        = 0.999
	lookaheadEpsilon      = 1e-8
	lookaheadMaxSteps     = 1000
	lookaheadTolerance    = 1e-6
)

// Lookahead minimizuje konfigurisanu benchmark funkciju koristeći Lookahead optimizator: unutrašnja Adam petlja
// ažurira brze težine, a svakih k koraka spore težine se pomeraju ka njima i sinhronizuju natrag
// problem.Point je početna tačka pretrage
func Lookahead(problem models.Problem) (result models.Result, err error) {

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

	learningRate := lookaheadLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	k := lookaheadK
	if v, ok := problem.Payload["k"].(float64); ok {
		k = int(v)
	}
	if k < 1 {
		k = 1
	}

	alpha := lookaheadAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	beta1 := lookaheadBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := lookaheadBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := lookaheadEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := lookaheadMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := lookaheadTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	slow := append([]float64(nil), problem.Point...)
	fast := append([]float64(nil), problem.Point...)
	m := make([]float64, len(fast))
	v := make([]float64, len(fast))
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(fast)
		if norm(gradient) < tolerance {
			break
		}

		t := steps + 1
		beta1T := math.Pow(beta1, float64(t))
		beta2T := math.Pow(beta2, float64(t))

		for i := range fast {
			m[i] = beta1*m[i] + (1-beta1)*gradient[i]
			v[i] = beta2*v[i] + (1-beta2)*gradient[i]*gradient[i]

			mHat := m[i] / (1 - beta1T)
			vHat := v[i] / (1 - beta2T)

			fast[i] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
		}

		if t%k == 0 {
			for i := range slow {
				slow[i] += alpha * (fast[i] - slow[i])
				fast[i] = slow[i]
			}
			for i := range m {
				m[i] = 0
				v[i] = 0
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, slow, fn.Evaluate(slow), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, slow, fn.Evaluate(slow), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "lookahead",
		Point:      slow,
		Value:      fn.Evaluate(slow),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
