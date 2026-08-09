package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	ftrlAlpha     = 0.1
	ftrlBeta      = 1.0
	ftrlL2        = 0.01
	ftrlMaxSteps  = 1000
	ftrlTolerance = 1e-6
	ftrlStepClip  = 1.0
)

// Ftrl minimizuje konfigurisanu benchmark funkciju koristeći Follow-The-Regularized-Leader: umesto inkrementalnog
// pomeraja, pozicija se svaki korak DIREKTNO IZRAČUNA iz akumuliranih gradijentnih statistika z i n — suštinski
// drugačije od svih ostalih solvera ovde koji pomeraju poziciju inkrementalno
// problem.Point je početna tačka pretrage
func Ftrl(problem models.Problem) (result models.Result, err error) {

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

	alpha := ftrlAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	betaFtrl := ftrlBeta
	if v, ok := problem.Payload["beta_ftrl"].(float64); ok {
		betaFtrl = v
	}

	l2 := ftrlL2
	if v, ok := problem.Payload["l2"].(float64); ok {
		l2 = v
	}

	maxSteps := ftrlMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := ftrlTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	z := make([]float64, dimensions)
	n := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		xNew := make([]float64, dimensions)
		for d := range xNew {
			sigma := (math.Sqrt(n[d]+grad[d]*grad[d]) - math.Sqrt(n[d])) / alpha
			z[d] += grad[d] - sigma*x[d]
			n[d] += grad[d] * grad[d]
			xNew[d] = -z[d] / ((betaFtrl+math.Sqrt(n[d]))/alpha + l2)
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = xNew[d] - x[d]
		}
		delta = clipStep(delta, ftrlStepClip)
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
		Method:     "ftrl",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
