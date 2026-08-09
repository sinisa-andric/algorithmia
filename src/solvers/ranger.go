package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	rangerLearningRate   = 0.1
	rangerBeta1          = 0.95
	rangerBeta2          = 0.999
	rangerEpsilon        = 1e-8
	rangerKLookahead     = 5
	rangerAlphaLookahead = 0.5
	rangerMaxSteps       = 1000
	rangerTolerance      = 1e-6
	rangerStepClip       = 1.0
)

// Ranger minimizuje konfigurisanu benchmark funkciju koristeći Ranger: unutrašnji RAdam-like optimizator (sa
// rektifikacijom varijanse kao radam.go) omotan Lookahead mehanizmom koji svakih k_lookahead koraka pomera sporu
// kopiju pozicije ka brzoj i sinhronizuje ih — za razliku od lookahead.go čiji je unutrašnji optimizator običan
// Adam bez rektifikacije varijanse
// problem.Point je početna tačka pretrage
func Ranger(problem models.Problem) (result models.Result, err error) {

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

	learningRate := rangerLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := rangerBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := rangerBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := rangerEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	kLookahead := rangerKLookahead
	if v, ok := problem.Payload["k_lookahead"].(float64); ok {
		kLookahead = int(v)
	}
	if kLookahead < 1 {
		kLookahead = 1
	}

	alphaLookahead := rangerAlphaLookahead
	if v, ok := problem.Payload["alpha_lookahead"].(float64); ok {
		alphaLookahead = v
	}

	maxSteps := rangerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := rangerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	rhoInf := 2/(1-beta2) - 1

	slow := append([]float64(nil), problem.Point...)
	fast := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)

	best := append([]float64(nil), fast...)
	bestValue := fn.Evaluate(fast)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, fast, proximalGradEps)
		t := float64(steps + 1)
		beta1T := math.Pow(beta1, t)
		beta2T := math.Pow(beta2, t)
		rhoT := rhoInf - 2*t*beta2T/(1-beta2T)

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			mHat := m[d] / (1 - beta1T)

			if rhoT > 4 {
				vHat := math.Sqrt(s[d] / (1 - beta2T))
				r := math.Sqrt((rhoT - 4) * (rhoT - 2) * rhoInf / ((rhoInf - 4) * (rhoInf - 2) * rhoT))
				delta[d] = -learningRate * r * mHat / (vHat + epsilon)
			} else {
				delta[d] = -learningRate * mHat
			}
		}
		delta = clipStep(delta, rangerStepClip)
		for d := range fast {
			fast[d] += delta[d]
		}

		if (steps+1)%kLookahead == 0 {
			for d := range slow {
				slow[d] += alphaLookahead * (fast[d] - slow[d])
				fast[d] = slow[d]
			}
		}

		value := fn.Evaluate(fast)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), fast...)
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
		Method:     "ranger",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
