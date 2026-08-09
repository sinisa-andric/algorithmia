package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	haltonSequenceSearchNumSamples = 500
	haltonSequenceSearchLower      = -5.0
	haltonSequenceSearchUpper      = 5.0
)

// haltonPrimeBases su baze korišćene za Van der Corput konstrukciju po dimenziji (dovoljno prostih brojeva za
// razumnu dimenzionalnost; testirano na n=2 gde se koriste baze 2 i 3)
var haltonPrimeBases = []int{2, 3, 5, 7, 11, 13}

// haltonVdc računa Van der Corput vrednost indeksa u zadatoj bazi: cifre indeksa u toj bazi se obrnu posle tačke
func haltonVdc(index int, base int) float64 {

	result := 0.0
	f := 1.0 / float64(base)
	for index > 0 {
		result += f * float64(index%base)
		index /= base
		f /= float64(base)
	}

	return result
}

// HaltonSequenceSearch minimizuje konfigurisanu benchmark funkciju uzorkovanjem preko Halton kvazi-nasumičnog
// niza niske diskrepancije (Van der Corput konstrukcija po bazama 2,3,...) umesto uniformno nasumičnog
// uzorkovanja kao random_search.go — kvazi-nasumičan niz ravnomernije pokriva prostor pretrage
// problem.Point određuje samo dimenziju pretrage
func HaltonSequenceSearch(problem models.Problem) (result models.Result, err error) {

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

	numSamples := haltonSequenceSearchNumSamples
	if v, ok := problem.Payload["num_samples"].(float64); ok {
		numSamples = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	lower, upper := haltonSequenceSearchLower, haltonSequenceSearchUpper

	var best []float64
	bestValue := math.Inf(1)

	steps := 0
	for i := 1; i <= numSamples; i++ {
		point := make([]float64, dimensions)
		for d := range point {
			point[d] = lower + haltonVdc(i, haltonPrimeBases[d%len(haltonPrimeBases)])*(upper-lower)
		}
		if value := fn.Evaluate(point); value < bestValue {
			bestValue = value
			best = point
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		steps++
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "halton_sequence_search",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
