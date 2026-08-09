package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	frankWolfeBoxMaxSteps  = 1000
	frankWolfeBoxTolerance = 1e-6
	frankWolfeBoxStepClip  = 1.0
	frankWolfeBoxLower     = -5.0
	frankWolfeBoxUpper     = 5.0
)

// FrankWolfeBox minimizuje konfigurisanu benchmark funkciju koristeći Frank-Wolfe (Conditional Gradient) metod,
// adaptiran na box ograničenje [-5,5]^n (koje JESTE konveksan skup, pa je primena originalnog Frank-Wolfe metoda
// nad njim legitimna, iako metod originalno radi nad proizvoljnim konveksnim skupom). DEFINIŠUĆA karakteristika
// Frank-Wolfe-a: pravac svakog koraka je uvek ka ĆOŠKU dozvoljenog skupa (linearni minimizator gradijenta nad
// box-om), ne ka lokalnom spustu kao svi ostali solveri u ovom modulu; korak je konveksna kombinacija ka tom ćošku
// sa opadajućom težinom gamma=2/(k+2)
// problem.Point je početna tačka pretrage (projektovana u [-5,5] ako je van opsega)
func FrankWolfeBox(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := frankWolfeBoxMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := frankWolfeBoxTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := make([]float64, dimensions)
	for d := range x {
		x[d] = clamp(problem.Point[d], frankWolfeBoxLower, frankWolfeBoxUpper)
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		s := make([]float64, dimensions)
		for d := range s {
			if grad[d] > 0 {
				s[d] = frankWolfeBoxLower
			} else {
				s[d] = frankWolfeBoxUpper
			}
		}

		gamma := 2.0 / (float64(steps) + 2.0)
		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = gamma * (s[d] - x[d])
		}
		delta = clipStep(delta, frankWolfeBoxStepClip)
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
		Method:     "frank_wolfe_box",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
