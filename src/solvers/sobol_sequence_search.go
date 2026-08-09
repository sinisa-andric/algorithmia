package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	sobolSequenceSearchNumSamples = 500
	sobolSequenceSearchLower      = -5.0
	sobolSequenceSearchUpper      = 5.0
)

// sobolPrimeBases: NAPOMENA — prava Sobol sekvenca zahteva specijalizovane "direction numbers" po dimenziji, što
// je van dometa ovog sistema. Ovo je POJEDNOSTAVLJENA aproksimacija: Van der Corput (Halton-stil) konstrukcija sa
// DRUGIM parom baza (2,5) nego halton_sequence_search.go (2,3), da bi se dobio drugačiji niz niske diskrepancije
var sobolPrimeBases = []int{2, 5, 7, 11, 13, 17}

// SobolSequenceSearch minimizuje konfigurisanu benchmark funkciju uzorkovanjem preko pojednostavljene Sobol-like
// kvazi-nasumične sekvence (Van der Corput konstrukcija sa bazama 2,5,... — vidi napomenu uz sobolPrimeBases)
// problem.Point određuje samo dimenziju pretrage
func SobolSequenceSearch(problem models.Problem) (result models.Result, err error) {

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

	numSamples := sobolSequenceSearchNumSamples
	if v, ok := problem.Payload["num_samples"].(float64); ok {
		numSamples = int(v)
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	lower, upper := sobolSequenceSearchLower, sobolSequenceSearchUpper

	var best []float64
	bestValue := math.Inf(1)

	steps := 0
	for i := 1; i <= numSamples; i++ {
		point := make([]float64, dimensions)
		for d := range point {
			point[d] = lower + haltonVdc(i, sobolPrimeBases[d%len(sobolPrimeBases)])*(upper-lower)
		}
		if value := fn.Evaluate(point); value < bestValue {
			bestValue = value
			best = point
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
		}
		steps++
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "sobol_sequence_search",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
