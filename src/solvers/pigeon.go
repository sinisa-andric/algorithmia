package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	pigeonPopulation = 20
	pigeonMaxSteps   = 1000
	pigeonTolerance  = 1e-6
	pigeonRangeLow   = -5.0
	pigeonRangeHigh  = 5.0
)

// Pigeon minimizuje konfigurisanu benchmark funkciju koristeći Pigeon Inspired Optimization: u prvoj polovini
// izvršavanja golubovi koriste operator mape i kompasa (kretanje pod uticajem brzine ka najboljem),
// u drugoj polovini koriste operator orijentira (kretanje ka centru boljih)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Pigeon(problem models.Problem) (result models.Result, err error) {

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

	population := pigeonPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := pigeonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := pigeonTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sortByValue := func(points [][]float64, vals []float64) {
		idx := make([]int, len(points))
		for i := range idx {
			idx[i] = i
		}
		for i := 1; i < len(idx); i++ {
			for j := i; j > 0 && vals[idx[j]] < vals[idx[j-1]]; j-- {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			}
		}
		sortedPoints := make([][]float64, len(points))
		sortedVals := make([]float64, len(points))
		for i, k := range idx {
			sortedPoints[i] = points[k]
			sortedVals[i] = vals[k]
		}
		copy(points, sortedPoints)
		copy(vals, sortedVals)
	}

	pigeons := make([][]float64, population)
	velocity := make([][]float64, population)
	values := make([]float64, population)
	for i := range pigeons {
		pigeon := make([]float64, dimensions)
		for d := range pigeon {
			pigeon[d] = pigeonRangeLow + rand.Float64()*(pigeonRangeHigh-pigeonRangeLow)
		}
		pigeons[i] = pigeon
		velocity[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(pigeon)
	}

	best := append([]float64(nil), pigeons[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), pigeons[i]...)
		}
	}

	const mapFactor = 0.2

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		if float64(steps) < float64(maxSteps)/2 {
			for i := range pigeons {
				for d := range pigeons[i] {
					velocity[i][d] = velocity[i][d]*math.Exp(-mapFactor*float64(steps)) + rand.Float64()*(best[d]-pigeons[i][d])
					pigeons[i][d] = clamp(pigeons[i][d]+velocity[i][d], pigeonRangeLow, pigeonRangeHigh)
				}
				values[i] = fn.Evaluate(pigeons[i])
			}
		} else {
			sortByValue(pigeons, values)

			half := max(1, population/2)
			center := make([]float64, dimensions)
			for i := range half {
				for d := range center {
					center[d] += pigeons[i][d]
				}
			}
			for d := range center {
				center[d] /= float64(half)
			}

			for i := range pigeons {
				for d := range pigeons[i] {
					pigeons[i][d] = clamp(pigeons[i][d]+rand.Float64()*(center[d]-pigeons[i][d]), pigeonRangeLow, pigeonRangeHigh)
				}
				values[i] = fn.Evaluate(pigeons[i])
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), pigeons[i]...)
			}
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
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "pigeon",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
