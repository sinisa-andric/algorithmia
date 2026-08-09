package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	goshawkPopulation = 20
	goshawkMaxSteps   = 1000
	goshawkTolerance  = 1e-6
	goshawkRangeLow   = -5.0
	goshawkRangeHigh  = 5.0
)

// Goshawk minimizuje konfigurisanu benchmark funkciju koristeći Northern Goshawk Optimization: jastreb u fazi
// identifikacije plena poredi se sa nasumičnim drugim jastrebom (kreće se ka njemu ako je bolji, inače se udaljava),
// uz slabu vezu ka best-u da roj ne kolabira, a u fazi progona sužava pretragu oko najboljeg rešenja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Goshawk(problem models.Problem) (result models.Result, err error) {

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

	population := goshawkPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := goshawkMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := goshawkTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	randomIndexExcept := func(exclude int) int {
		if population < 2 {
			return exclude
		}
		j := rand.IntN(population)
		for j == exclude {
			j = rand.IntN(population)
		}
		return j
	}

	goshawks := make([][]float64, population)
	values := make([]float64, population)
	for i := range goshawks {
		goshawk := make([]float64, dimensions)
		for d := range goshawk {
			goshawk[d] = goshawkRangeLow + rand.Float64()*(goshawkRangeHigh-goshawkRangeLow)
		}
		goshawks[i] = goshawk
		values[i] = fn.Evaluate(goshawk)
	}

	best := append([]float64(nil), goshawks[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), goshawks[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range goshawks {
			if rand.Float64() < 0.5 {
				r := randomIndexExcept(i)
				// poređenje sa nasumičnim jastrebom je čisto peer-bazirano — dodata je slaba veza ka
				// best-u da roj ne kolabira u proizvoljnu tačku pre nego što ubode pravi minimum
				sign := 1.0
				if values[r] >= values[i] {
					sign = -1.0
				}
				for d := range goshawks[i] {
					goshawks[i][d] = goshawks[i][d] + sign*rand.Float64()*(goshawks[r][d]-goshawks[i][d]) + rand.Float64()*(best[d]-goshawks[i][d])*0.1
				}
			} else {
				radius := 0.5 * (1 - float64(steps)/float64(maxSteps))
				for d := range goshawks[i] {
					goshawks[i][d] = best[d] + (rand.Float64()*2-1)*radius
				}
			}

			for d := range goshawks[i] {
				goshawks[i][d] = clamp(goshawks[i][d], goshawkRangeLow, goshawkRangeHigh)
			}
			values[i] = fn.Evaluate(goshawks[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), goshawks[i]...)
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
		Method:     "goshawk",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
