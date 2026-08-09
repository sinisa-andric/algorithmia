package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	dungBeetlePopulation = 20
	dungBeetleMaxSteps   = 1000
	dungBeetleTolerance  = 1e-6
	dungBeetleRangeLow   = -5.0
	dungBeetleRangeHigh  = 5.0
)

// DungBeetle minimizuje konfigurisanu benchmark funkciju koristeći Dung Beetle Optimizer: većina bubara gura loptu
// udaljavajući se od najgoreg i (povremeno) približavajući se najboljem rešenju,
// dok manjina umesto toga navigira direktno prema kombinaciji dva nasumična bubara
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func DungBeetle(problem models.Problem) (result models.Result, err error) {

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

	population := dungBeetlePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := dungBeetleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dungBeetleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	beetles := make([][]float64, population)
	values := make([]float64, population)
	for i := range beetles {
		beetle := make([]float64, dimensions)
		for d := range beetle {
			beetle[d] = dungBeetleRangeLow + rand.Float64()*(dungBeetleRangeHigh-dungBeetleRangeLow)
		}
		beetles[i] = beetle
		values[i] = fn.Evaluate(beetle)
	}

	best := append([]float64(nil), beetles[0]...)
	bestValue := values[0]
	worst := append([]float64(nil), beetles[0]...)
	worstValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), beetles[i]...)
		}
		if v > worstValue {
			worstValue = v
			worst = append([]float64(nil), beetles[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range beetles {
			if rand.Float64() < 0.5 {
				for d := range beetles[i] {
					delta := rand.Float64() * (beetles[i][d] - worst[d])
					beetles[i][d] = best[d] + delta*math.Exp(-float64(steps)/float64(maxSteps))
				}
			} else {
				r1Idx := rand.IntN(population)
				for r1Idx == i {
					r1Idx = rand.IntN(population)
				}
				r2Idx := rand.IntN(population)
				for r2Idx == i || r2Idx == r1Idx {
					r2Idx = rand.IntN(population)
				}
				r1 := beetles[r1Idx]
				r2 := beetles[r2Idx]
				for d := range beetles[i] {
					beetles[i][d] = beetles[i][d] + rand.Float64()*(best[d]-beetles[i][d]) + rand.Float64()*(r1[d]-r2[d])
				}
			}

			for d := range beetles[i] {
				beetles[i][d] = clamp(beetles[i][d], dungBeetleRangeLow, dungBeetleRangeHigh)
			}

			values[i] = fn.Evaluate(beetles[i])
		}

		bestValue = values[0]
		best = append([]float64(nil), beetles[0]...)
		worstValue = values[0]
		worst = append([]float64(nil), beetles[0]...)
		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), beetles[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
			if v > worstValue {
				worstValue = v
				worst = append([]float64(nil), beetles[i]...)
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
		Method:     "dung_beetle",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
