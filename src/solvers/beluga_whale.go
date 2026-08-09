package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	belugaWhalePopulation = 20
	belugaWhaleMaxSteps   = 1000
	belugaWhaleTolerance  = 1e-6
	belugaWhaleRangeLow   = -5.0
	belugaWhaleRangeHigh  = 5.0
	belugaWhaleBf         = 0.1
)

// BelugaWhale minimizuje konfigurisanu benchmark funkciju koristeći Beluga Whale Optimization:
// svaki kit ili pliva i hrani se krećući se duž razlike dva nasumična kita, ili prolazi kroz fallout fazu gde
// konvergira ka najboljem rešenju; svakih deset koraka najgori deo populacije se zamenjuje nasumičnim jedinkama
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func BelugaWhale(problem models.Problem) (result models.Result, err error) {

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

	population := belugaWhalePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := belugaWhaleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := belugaWhaleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	whales := make([][]float64, population)
	values := make([]float64, population)
	for i := range whales {
		whale := make([]float64, dimensions)
		for d := range whale {
			whale[d] = belugaWhaleRangeLow + rand.Float64()*(belugaWhaleRangeHigh-belugaWhaleRangeLow)
		}
		whales[i] = whale
		values[i] = fn.Evaluate(whale)
	}

	best := append([]float64(nil), whales[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), whales[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range whales {
			if rand.Float64() < 0.5 {
				r1 := whales[rand.IntN(population)]
				r2 := whales[rand.IntN(population)]
				for d := range whales[i] {
					whales[i][d] += rand.Float64() * (r1[d] - r2[d])
				}
			} else {
				c1 := rand.Float64() * (1 - float64(steps)/float64(maxSteps))
				c2 := rand.Float64() * (1 - float64(steps)/float64(maxSteps))
				for d := range whales[i] {
					whales[i][d] = (1-c1)*best[d] + c2*rand.Float64()*(best[d]-whales[i][d])
				}
			}

			for d := range whales[i] {
				whales[i][d] = clamp(whales[i][d], belugaWhaleRangeLow, belugaWhaleRangeHigh)
			}

			values[i] = fn.Evaluate(whales[i])
		}

		if steps%10 == 0 {
			order := make([]int, population)
			for i := range order {
				order[i] = i
			}
			sort.Slice(order, func(a, b int) bool {
				return values[order[a]] > values[order[b]]
			})

			replaceCount := int(belugaWhaleBf * float64(population))
			for k := 0; k < replaceCount; k++ {
				idx := order[k]
				whale := make([]float64, dimensions)
				for d := range whale {
					whale[d] = belugaWhaleRangeLow + rand.Float64()*(belugaWhaleRangeHigh-belugaWhaleRangeLow)
				}
				whales[idx] = whale
				values[idx] = fn.Evaluate(whale)
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), whales[i]...)
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
		Method:     "beluga_whale",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
