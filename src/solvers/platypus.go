package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	platypusPopulation = 20
	platypusMaxSteps   = 1000
	platypusTolerance  = 1e-6
	platypusRangeLow   = -5.0
	platypusRangeHigh  = 5.0
)

// Platypus minimizuje konfigurisanu benchmark funkciju koristeći Platypus Optimizer: čudnovati kljunaš naizmenično
// roni (lokalna pretraga unutar opsega elektrolokacije koji se sužava tokom izvršavanja) ili se kreće ka najboljem
// rešenju putem elektro-signala
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Platypus(problem models.Problem) (result models.Result, err error) {

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

	population := platypusPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := platypusMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := platypusTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	platypuses := make([][]float64, population)
	values := make([]float64, population)
	for i := range platypuses {
		platypus := make([]float64, dimensions)
		for d := range platypus {
			platypus[d] = platypusRangeLow + rand.Float64()*(platypusRangeHigh-platypusRangeLow)
		}
		platypuses[i] = platypus
		values[i] = fn.Evaluate(platypus)
	}

	best := append([]float64(nil), platypuses[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), platypuses[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		detectionRadius := 5 * (1 - float64(steps)/float64(maxSteps))

		for i := range platypuses {
			if rand.Float64() < 0.5 {
				for d := range platypuses[i] {
					platypuses[i][d] = platypuses[i][d] + (rand.Float64()*2-1)*detectionRadius*0.3
				}
			} else {
				for d := range platypuses[i] {
					platypuses[i][d] = platypuses[i][d] + rand.Float64()*(best[d]-platypuses[i][d])
				}
			}

			for d := range platypuses[i] {
				platypuses[i][d] = clamp(platypuses[i][d], platypusRangeLow, platypusRangeHigh)
			}
			values[i] = fn.Evaluate(platypuses[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), platypuses[i]...)
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
		Method:     "platypus",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
