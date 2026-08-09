package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	powellConjugateDirectionMaxSteps  = 1000
	powellConjugateDirectionTolerance = 1e-6
	powellConjugateDirectionStepClip  = 1.0
)

// powellLineSearchSteps su kandidati za veličinu 1D linijske pretrage duž jednog pravca (probaju se oba predznaka)
var powellLineSearchSteps = []float64{1.0, 0.5, 0.25, 0.1, 0.05}

// powellLineSearch1D vraća najbolju tačku duž linije x+t*d za t u ±powellLineSearchSteps, ili x nepromenjeno ako
// nijedan kandidat ne poboljšava vrednost
func powellLineSearch1D(f func([]float64) float64, x, d []float64) []float64 {

	best := append([]float64(nil), x...)
	bestValue := f(x)
	for _, t := range powellLineSearchSteps {
		for _, sign := range []float64{1.0, -1.0} {
			step := make([]float64, len(x))
			for i := range step {
				step[i] = sign * t * d[i]
			}
			step = clipStep(step, powellConjugateDirectionStepClip)
			trial := make([]float64, len(x))
			for i := range trial {
				trial[i] = x[i] + step[i]
			}
			if v := f(trial); v < bestValue {
				bestValue = v
				best = trial
			}
		}
	}

	return best
}

// PowellConjugateDirection minimizuje konfigurisanu benchmark funkciju koristeći Powell-ov metod konjugovanih
// pravaca: čisto derivative-free (BEZ gradijenta, za razliku od CG varijanti) — svaki ciklus radi 1D linijsku
// pretragu duž svakog trenutnog pravca, zatim zamenjuje najstariji pravac ukupnim pomerajem celog ciklusa
// (Powell-ova zamena pravca) i radi dodatnu pretragu duž tog novog pravca
// problem.Point je početna tačka pretrage
func PowellConjugateDirection(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := powellConjugateDirectionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := powellConjugateDirectionTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := append([]float64(nil), problem.Point...)

	directions := make([][]float64, dimensions)
	for i := range directions {
		directions[i] = make([]float64, dimensions)
		directions[i][i] = 1.0
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		xStart := append([]float64(nil), x...)
		for _, d := range directions {
			x = powellLineSearch1D(fn.Evaluate, x, d)
		}

		newDirection := make([]float64, dimensions)
		for i := range newDirection {
			newDirection[i] = x[i] - xStart[i]
		}

		// Powell-ova zamena pravca: ukloni najstariji (indeks 0), dodaj novi pravac na kraj
		directions = append(append([][]float64{}, directions[1:]...), newDirection)
		x = powellLineSearch1D(fn.Evaluate, x, newDirection)

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
		Method:     "powell_conjugate_direction",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
