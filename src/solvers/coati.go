package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	coatiPopulation = 20
	coatiMaxSteps   = 1000
	coatiTolerance  = 1e-6
	coatiRangeLow   = -5.0
	coatiRangeHigh  = 5.0
)

// Coati minimizuje konfigurisanu benchmark funkciju koristeći Coati Optimization Algorithm: svaki koati prvo napada
// nasumičnu iguanu — krećući se ka njoj ako je bolja, ili udaljavajući se ako nije (eksploracija) —
// a zatim se penje na drvo skačući u interval oko najboljeg rešenja koji se sužava tokom izvršavanja (eksploatacija)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Coati(problem models.Problem) (result models.Result, err error) {

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

	population := coatiPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := coatiMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := coatiTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	coatis := make([][]float64, population)
	values := make([]float64, population)
	for i := range coatis {
		coati := make([]float64, dimensions)
		for d := range coati {
			coati[d] = coatiRangeLow + rand.Float64()*(coatiRangeHigh-coatiRangeLow)
		}
		coatis[i] = coati
		values[i] = fn.Evaluate(coati)
	}

	best := append([]float64(nil), coatis[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), coatis[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		factor := 1 - float64(steps)/float64(maxSteps)

		for i := range coatis {
			iguana := make([]float64, dimensions)
			for d := range iguana {
				iguana[d] = coatiRangeLow + rand.Float64()*(coatiRangeHigh-coatiRangeLow)
			}
			if fn.Evaluate(iguana) < fn.Evaluate(coatis[i]) {
				for d := range coatis[i] {
					coatis[i][d] += rand.Float64() * (iguana[d] - coatis[i][d])
				}
			} else {
				for d := range coatis[i] {
					coatis[i][d] += rand.Float64() * (coatis[i][d] - iguana[d])
				}
			}
			for d := range coatis[i] {
				coatis[i][d] = clamp(coatis[i][d], coatiRangeLow, coatiRangeHigh)
			}

			for d := range coatis[i] {
				lb := best[d] - (best[d]-coatis[i][d])*factor
				ub := best[d] + (best[d]-coatis[i][d])*factor
				if lb > ub {
					lb, ub = ub, lb
				}
				coatis[i][d] = lb + rand.Float64()*(ub-lb)
			}

			for d := range coatis[i] {
				coatis[i][d] = clamp(coatis[i][d], coatiRangeLow, coatiRangeHigh)
			}

			values[i] = fn.Evaluate(coatis[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), coatis[i]...)
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
		Method:     "coati",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
