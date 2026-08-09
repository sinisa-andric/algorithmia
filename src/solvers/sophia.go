package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	sophiaLearningRate = 0.1
	sophiaBeta1        = 0.965
	sophiaBeta2        = 0.99
	sophiaEpsilon      = 1e-12
	sophiaRhoClip      = 0.04
	sophiaMaxSteps     = 1000
	sophiaTolerance    = 1e-6
	sophiaStepClip     = 1.0
)

// Sophia minimizuje konfigurisanu benchmark funkciju koristeći Sophia optimizator: koristi PRAVU numeričku
// dijagonalu Hesijana (ne secant aproksimaciju kao apollo.go) uglađenu EMA-om, sa eksplicitnim element-wise clip-om
// primenjenim na sam update PRE množenja sa learning_rate-om — razlika od apollo.go koje nema clip na update i
// koristi secant informaciju umesto prave druge izvod
// problem.Point je početna tačka pretrage
func Sophia(problem models.Problem) (result models.Result, err error) {

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

	learningRate := sophiaLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := sophiaBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := sophiaBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := sophiaEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	rhoClip := sophiaRhoClip
	if v, ok := problem.Payload["rho_clip"].(float64); ok {
		rhoClip = v
	}

	maxSteps := sophiaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sophiaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	hessDiag := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for d := range m {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
		}

		hess := numHessDiag(fn.Evaluate, x, hessDiagEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			hessDiag[d] = beta2*hessDiag[d] + (1-beta2)*hess[d]
			update := clamp(m[d]/math.Max(hessDiag[d], epsilon), -rhoClip, rhoClip)
			delta[d] = -learningRate * update
		}
		delta = clipStep(delta, sophiaStepClip)
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
		Method:     "sophia",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
