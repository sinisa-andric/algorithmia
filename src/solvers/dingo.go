package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	dingoPopulation = 20
	dingoMaxSteps   = 1000
	dingoTolerance  = 1e-6
	dingoRangeLow   = -5.0
	dingoRangeHigh  = 5.0
)

// Dingo minimizuje konfigurisanu benchmark funkciju koristeći Dingo Optimization Algorithm: dingo ili lovi u grupi
// kombinujući smer ka najboljem rešenju sa razlikom dva nasumična dinga, ili napada direktno ekstrapolirajući
// poziciju preko najboljeg rešenja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Dingo(problem models.Problem) (result models.Result, err error) {

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

	population := dingoPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := dingoMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dingoTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	dingos := make([][]float64, population)
	values := make([]float64, population)
	for i := range dingos {
		dingo := make([]float64, dimensions)
		for d := range dingo {
			dingo[d] = dingoRangeLow + rand.Float64()*(dingoRangeHigh-dingoRangeLow)
		}
		dingos[i] = dingo
		values[i] = fn.Evaluate(dingo)
	}

	best := append([]float64(nil), dingos[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), dingos[i]...)
		}
	}

	twoRandomIndicesExcept := func(exclude int) (int, int) {
		if population < 3 {
			return exclude, exclude
		}
		r1 := rand.IntN(population)
		for r1 == exclude {
			r1 = rand.IntN(population)
		}
		r2 := rand.IntN(population)
		for r2 == exclude || r2 == r1 {
			r2 = rand.IntN(population)
		}
		return r1, r2
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range dingos {
			p := rand.Float64()
			if p < 0.5 {
				r1, r2 := twoRandomIndicesExcept(i)
				for d := range dingos[i] {
					dingos[i][d] = dingos[i][d] + rand.Float64()*(best[d]-dingos[i][d]) + rand.Float64()*(dingos[r1][d]-dingos[r2][d])
				}
			} else {
				for d := range dingos[i] {
					dingos[i][d] = best[d] + rand.Float64()*(best[d]-dingos[i][d])
				}
			}

			for d := range dingos[i] {
				dingos[i][d] = clamp(dingos[i][d], dingoRangeLow, dingoRangeHigh)
			}
			values[i] = fn.Evaluate(dingos[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), dingos[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
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
		Method:     "dingo",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
