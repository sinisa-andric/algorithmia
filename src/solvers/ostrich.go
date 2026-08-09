package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	ostrichPopulation = 20
	ostrichMaxSteps   = 1000
	ostrichTolerance  = 1e-6
	ostrichRangeLow   = -5.0
	ostrichRangeHigh  = 5.0
)

// Ostrich minimizuje konfigurisanu benchmark funkciju koristeći Ostrich Optimization Algorithm: noj se nasumično
// opredeljuje za jedno od tri ponašanja — ispašu (lokalna pretraga oko najboljeg),
// budnost (kretanje ka centru jata) ili bekstvo (nasumičan veliki skok)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Ostrich(problem models.Problem) (result models.Result, err error) {

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

	population := ostrichPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := ostrichMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := ostrichTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	ostriches := make([][]float64, population)
	values := make([]float64, population)
	for i := range ostriches {
		ostrich := make([]float64, dimensions)
		for d := range ostrich {
			ostrich[d] = ostrichRangeLow + rand.Float64()*(ostrichRangeHigh-ostrichRangeLow)
		}
		ostriches[i] = ostrich
		values[i] = fn.Evaluate(ostrich)
	}

	best := append([]float64(nil), ostriches[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), ostriches[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		center := make([]float64, dimensions)
		for _, o := range ostriches {
			for d := range center {
				center[d] += o[d]
			}
		}
		for d := range center {
			center[d] /= float64(population)
		}

		for i := range ostriches {
			p := rand.Float64()
			switch {
			case p < 0.4:
				for d := range ostriches[i] {
					ostriches[i][d] = best[d] + (rand.Float64()*2-1)*0.3
				}
			case p < 0.7:
				for d := range ostriches[i] {
					ostriches[i][d] = ostriches[i][d] + rand.Float64()*(center[d]-ostriches[i][d])
				}
			default:
				for d := range ostriches[i] {
					ostriches[i][d] = ostriches[i][d] + (rand.Float64()*2-1)*2.0
				}
			}

			for d := range ostriches[i] {
				ostriches[i][d] = clamp(ostriches[i][d], ostrichRangeLow, ostrichRangeHigh)
			}
			values[i] = fn.Evaluate(ostriches[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), ostriches[i]...)
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
		Method:     "ostrich",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
