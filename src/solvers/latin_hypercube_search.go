package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	latinHypercubeSearchNumSamples = 500
	latinHypercubeSearchLower      = -5.0
	latinHypercubeSearchUpper      = 5.0
)

// LatinHypercubeSearch minimizuje konfigurisanu benchmark funkciju Latin Hypercube uzorkovanjem: svaka dimenzija
// se nezavisno deli na num_samples jednakih segmenata koji se nasumično permutuju, i uzima se po jedna nasumična
// tačka iz svakog segmenta — GARANTUJE da je svaka 1D projekcija ravnomerno pokrivena, za razliku od
// random_search.go i halton/sobol koji to eksplicitno ne stratifikuju po dimenziji
// problem.Point određuje samo dimenziju pretrage
func LatinHypercubeSearch(problem models.Problem) (result models.Result, err error) {

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

	numSamples := latinHypercubeSearchNumSamples
	if v, ok := problem.Payload["num_samples"].(float64); ok {
		numSamples = int(v)
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	lower, upper := latinHypercubeSearchLower, latinHypercubeSearchUpper
	segmentWidth := (upper - lower) / float64(numSamples)

	columns := make([][]float64, dimensions)
	for d := range columns {
		perm := make([]int, numSamples)
		for i := range perm {
			perm[i] = i
		}
		for i := numSamples - 1; i > 0; i-- {
			j := rand.IntN(i + 1)
			perm[i], perm[j] = perm[j], perm[i]
		}

		col := make([]float64, numSamples)
		for i, segment := range perm {
			segStart := lower + float64(segment)*segmentWidth
			col[i] = segStart + rand.Float64()*segmentWidth
		}
		columns[d] = col
	}

	var best []float64
	bestValue := math.Inf(1)

	steps := 0
	for i := 0; i < numSamples; i++ {
		point := make([]float64, dimensions)
		for d := range point {
			point[d] = columns[d][i]
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
		Method:     "latin_hypercube_search",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
