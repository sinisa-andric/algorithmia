package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	radamboundLearningRate = 0.1
	radamboundBeta1        = 0.9
	radamboundBeta2        = 0.999
	radamboundEpsilon      = 1e-8
	radamboundFinalLr      = 0.1
	radamboundGamma        = 0.001
	radamboundMaxSteps     = 1000
	radamboundTolerance    = 1e-6
	radamboundStepClip     = 1.0
)

// Radambound minimizuje konfigurisanu benchmark funkciju koristeći RAdamBound: standardna RAdam rektifikacija
// varijanse (rho_inf/rho_t kao radam.go) KOMBINOVANA sa dinamičkim bound-om na step_size u adabound.go stilu —
// razlika od oba: radam.go nema bound, adabound.go nema rektifikaciju varijanse
// problem.Point je početna tačka pretrage
func Radambound(problem models.Problem) (result models.Result, err error) {

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

	learningRate := radamboundLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := radamboundBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := radamboundBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := radamboundEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	finalLr := radamboundFinalLr
	if v, ok := problem.Payload["final_lr"].(float64); ok {
		finalLr = v
	}

	maxSteps := radamboundMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := radamboundTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	rhoInf := 2/(1-beta2) - 1

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		t := float64(steps + 1)
		beta1T := math.Pow(beta1, t)
		beta2T := math.Pow(beta2, t)
		rhoT := rhoInf - 2*t*beta2T/(1-beta2T)

		// gamma za dinamički bound nije eksplicitno naveden u specifikaciji ovog solvera (za razliku od
		// amsbound.go), pa se koristi fiksna interna konstanta umesto korisnički podesivog parametra
		lower := finalLr * (1 - 1/(radamboundGamma*t+1))
		upper := finalLr * (1 + 1/(radamboundGamma*t))

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			mHat := m[d] / (1 - beta1T)

			var rawStep float64
			if rhoT > 4 {
				vHat := math.Sqrt(s[d] / (1 - beta2T))
				r := math.Sqrt((rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT))
				rawStep = r * mHat / (vHat + epsilon)
			} else {
				rawStep = mHat
			}

			stepSize := clamp(learningRate, lower, upper)
			delta[d] = -stepSize * rawStep
		}
		delta = clipStep(delta, radamboundStepClip)
		for d := range x {
			x[d] += delta[d]
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
		Method:     "radambound",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
