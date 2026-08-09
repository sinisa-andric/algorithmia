package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	jellyfishPopulation = 20
	jellyfishMaxSteps   = 1000
	jellyfishTolerance  = 1e-6
	jellyfishRangeLow   = -5.0
	jellyfishRangeHigh  = 5.0
)

// Jellyfish minimizuje konfigurisanu benchmark funkciju koristeći Jellyfish Search Optimizer: kontrolni parametar
// izveden iz kosinusne funkcije bira između kretanja sa okeanskom strujom ka najboljem rešenju i kretanja unutar
// jata, koje je ili pasivno (nasumična perturbacija) ili aktivno (kretanje duž razlike sa boljom nasumičnom meduzom)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Jellyfish(problem models.Problem) (result models.Result, err error) {

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

	population := jellyfishPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := jellyfishMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := jellyfishTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	jellyfish := make([][]float64, population)
	values := make([]float64, population)
	for i := range jellyfish {
		j := make([]float64, dimensions)
		for d := range j {
			j[d] = jellyfishRangeLow + rand.Float64()*(jellyfishRangeHigh-jellyfishRangeLow)
		}
		jellyfish[i] = j
		values[i] = fn.Evaluate(j)
	}

	best := append([]float64(nil), jellyfish[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), jellyfish[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		ct := math.Abs(math.Cos(math.Pi / 2 * rand.Float64()))

		mean := make([]float64, dimensions)
		for _, j := range jellyfish {
			for d, v := range j {
				mean[d] += v / float64(population)
			}
		}

		for i := range jellyfish {
			if ct > 0.5 {
				for d := range jellyfish[i] {
					jellyfish[i][d] += rand.Float64() * (best[d] - math.Abs(mean[d])*jellyfish[i][d])
				}
			} else if rand.Float64() > 1-ct {
				for d := range jellyfish[i] {
					randPos := jellyfishRangeLow + rand.Float64()*(jellyfishRangeHigh-jellyfishRangeLow)
					jellyfish[i][d] += 0.1 * rand.Float64() * (randPos - jellyfish[i][d])
				}
			} else {
				r := jellyfish[rand.IntN(population)]
				better := fn.Evaluate(r) < fn.Evaluate(jellyfish[i])
				for d := range jellyfish[i] {
					var diff float64
					if better {
						diff = r[d] - jellyfish[i][d]
					} else {
						diff = jellyfish[i][d] - r[d]
					}
					jellyfish[i][d] += rand.Float64() * diff
				}
			}

			for d := range jellyfish[i] {
				jellyfish[i][d] = clamp(jellyfish[i][d], jellyfishRangeLow, jellyfishRangeHigh)
			}

			values[i] = fn.Evaluate(jellyfish[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), jellyfish[i]...)
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
		Method:     "jellyfish",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
