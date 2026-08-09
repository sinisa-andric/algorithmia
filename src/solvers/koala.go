package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	koalaPopulation = 20
	koalaMaxSteps   = 1000
	koalaTolerance  = 1e-6
	koalaRangeLow   = -5.0
	koalaRangeHigh  = 5.0
)

// Koala minimizuje konfigurisanu benchmark funkciju koristeći Koala Optimization Algorithm: koala se ili penje ka
// vrhu najboljeg stabla (eksploatacija koja jača tokom izvršavanja) ili prelazi na susedno stablo radi istraživanja
// prostora pretrage
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Koala(problem models.Problem) (result models.Result, err error) {

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

	population := koalaPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := koalaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := koalaTolerance
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

	koalas := make([][]float64, population)
	values := make([]float64, population)
	for i := range koalas {
		koala := make([]float64, dimensions)
		for d := range koala {
			koala[d] = koalaRangeLow + rand.Float64()*(koalaRangeHigh-koalaRangeLow)
		}
		koalas[i] = koala
		values[i] = fn.Evaluate(koala)
	}

	best := append([]float64(nil), koalas[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), koalas[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range koalas {
			if rand.Float64() < 0.5 {
				for d := range koalas[i] {
					koalas[i][d] = koalas[i][d] + rand.Float64()*(best[d]-koalas[i][d])*(1-float64(steps)/float64(maxSteps))
				}
			} else {
				r := koalas[randomIndexExcept(i)]
				for d := range koalas[i] {
					koalas[i][d] = koalas[i][d] + (rand.Float64()*2-1)*(r[d]-koalas[i][d])*0.5
				}
			}

			for d := range koalas[i] {
				koalas[i][d] = clamp(koalas[i][d], koalaRangeLow, koalaRangeHigh)
			}
			values[i] = fn.Evaluate(koalas[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), koalas[i]...)
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
		Method:     "koala",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
