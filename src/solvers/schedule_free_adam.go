package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	scheduleFreeAdamLearningRate = 0.1
	scheduleFreeAdamBeta1        = 0.9
	scheduleFreeAdamBeta2        = 0.999
	scheduleFreeAdamEpsilon      = 1e-8
	scheduleFreeAdamMaxSteps     = 1000
	scheduleFreeAdamTolerance    = 1e-6
	scheduleFreeAdamStepClip     = 1.0
)

// ScheduleFreeAdam minimizuje konfigurisanu benchmark funkciju koristeći Schedule-Free Adam: pomoćna tačka z je
// SOPSTVENA istrajna sekvenca koja svaki korak radi PUN (nedecajovan) Adam korak, a izveštena pozicija x je Polyak
// prosek (težina 1/(step+1)) te z sekvence — dve odvojene sekvence sa usrednjavanjem umesto eksplicitnog opadanja
// learning rate-a, otud "schedule-free"
// NAPOMENA: prva verzija je pogrešno računala z iznova iz TRENUTNOG x svakog koraka umesto iz sopstvene istorije,
// što je implicitno uvelo baš onu 1/t opadajuću stopu učenja koju schedule-free metoda treba da izbegne (izmereno:
// 5x veći budžet koraka je poboljšao vrednost za svega ~4% — harmonijski red, ne stvarna konvergencija)
// problem.Point je početna tačka pretrage
func ScheduleFreeAdam(problem models.Problem) (result models.Result, err error) {

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

	learningRate := scheduleFreeAdamLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := scheduleFreeAdamBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := scheduleFreeAdamBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := scheduleFreeAdamEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := scheduleFreeAdamMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := scheduleFreeAdamTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	z := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, z, proximalGradEps)
		t := float64(steps + 1)
		beta1T := math.Pow(beta1, t)
		beta2T := math.Pow(beta2, t)

		zDelta := make([]float64, dimensions)
		for d := range zDelta {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			mHat := m[d] / (1 - beta1T)
			sHat := s[d] / (1 - beta2T)
			zDelta[d] = -learningRate * mHat / (math.Sqrt(sHat) + epsilon)
		}
		zDelta = clipStep(zDelta, scheduleFreeAdamStepClip)
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
		Method:     "schedule_free_adam",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
