package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	harrisHawksPopulation = 20
	harrisHawksMaxSteps   = 1000
	harrisHawksTolerance  = 1e-6
	harrisHawksRangeLow   = -5.0
	harrisHawksRangeHigh  = 5.0
	harrisHawksLevyLambda = 1.5
)

// HarrisHawks minimizuje konfigurisanu benchmark funkciju koristeći Harris Hawks Optimization: energija plena opada
// tokom izvršavanja — dok je visoka jastrebovi istražuju oko roja ili nasumične pozicije,
// a kad padne prelaze na meko ili tvrdo opsedanje plena, po potrebi uz Levy-flight korake
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func HarrisHawks(problem models.Problem) (result models.Result, err error) {

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

	population := harrisHawksPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := harrisHawksMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := harrisHawksTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	hawks := make([][]float64, population)
	values := make([]float64, population)
	for i := range hawks {
		hawk := make([]float64, dimensions)
		for d := range hawk {
			hawk[d] = harrisHawksRangeLow + rand.Float64()*(harrisHawksRangeHigh-harrisHawksRangeLow)
		}
		hawks[i] = hawk
		values[i] = fn.Evaluate(hawk)
	}

	best := append([]float64(nil), hawks[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), hawks[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		mean := make([]float64, dimensions)
		for _, hawk := range hawks {
			for d, v := range hawk {
				mean[d] += v / float64(population)
			}
		}

		E0 := 2*rand.Float64() - 1

		for i := range hawks {
			e := 2 * E0 * (1 - float64(steps)/float64(maxSteps))
			absE := math.Abs(e)

			if absE >= 1 {
				if rand.Float64() >= 0.5 {
					q := hawks[rand.IntN(population)]
					for d := range hawks[i] {
						hawks[i][d] = q[d] - rand.Float64()*math.Abs(q[d]-2*rand.Float64()*hawks[i][d])
					}
				} else {
					for d := range hawks[i] {
						lb, ub := harrisHawksRangeLow, harrisHawksRangeHigh
						hawks[i][d] = (best[d] - mean[d]) - rand.Float64()*(lb+rand.Float64()*(ub-lb))
					}
				}
			} else {
				j := 2 * (1 - rand.Float64())
				r := rand.Float64()

				if r >= 0.5 {
					for d := range hawks[i] {
						hawks[i][d] = best[d] - e*math.Abs(best[d]-hawks[i][d])
					}
				} else {
					y := make([]float64, dimensions)
					for d := range y {
						y[d] = best[d] - e*math.Abs(j*best[d]-hawks[i][d])
					}
					z := make([]float64, dimensions)
					for d := range z {
						z[d] = y[d] + (rand.Float64()*2-1)*mantegnaLevy(harrisHawksLevyLambda)
					}
					if fn.Evaluate(y) < fn.Evaluate(hawks[i]) {
						hawks[i] = y
					} else {
						hawks[i] = z
					}
				}
			}

			for d := range hawks[i] {
				hawks[i][d] = clamp(hawks[i][d], harrisHawksRangeLow, harrisHawksRangeHigh)
			}

			values[i] = fn.Evaluate(hawks[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), hawks[i]...)
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
		Method:     "harris_hawks",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// clamp ograničava v na interval [lo, hi]
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
