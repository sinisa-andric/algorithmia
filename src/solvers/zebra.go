package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	zebraPopulation = 20
	zebraMaxSteps   = 1000
	zebraTolerance  = 1e-6
	zebraRangeLow   = -5.0
	zebraRangeHigh  = 5.0
)

// Zebra minimizuje konfigurisanu benchmark funkciju koristeći Zebra Optimization Algorithm: svaka zebra prvo traži
// hranu krećući se ka boljoj nasumičnoj zebri ili udaljavajući se od lošije,
// a zatim simulira odbranu od predatora nasumičnim skokom skaliranim sopstvenom pozicijom
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Zebra(problem models.Problem) (result models.Result, err error) {

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

	population := zebraPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := zebraMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := zebraTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	zebras := make([][]float64, population)
	values := make([]float64, population)
	for i := range zebras {
		zebra := make([]float64, dimensions)
		for d := range zebra {
			zebra[d] = zebraRangeLow + rand.Float64()*(zebraRangeHigh-zebraRangeLow)
		}
		zebras[i] = zebra
		values[i] = fn.Evaluate(zebra)
	}

	best := append([]float64(nil), zebras[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), zebras[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range zebras {
			r := zebras[rand.IntN(population)]
			if fn.Evaluate(r) < fn.Evaluate(zebras[i]) {
				for d := range zebras[i] {
					zebras[i][d] += rand.Float64() * (r[d] - zebras[i][d])
				}
			} else {
				for d := range zebras[i] {
					zebras[i][d] -= rand.Float64() * (r[d] - zebras[i][d])
				}
			}

			ps := 1.0
			if rand.Float64() < 0.5 {
				ps = -1
			}
			for d := range zebras[i] {
				mz := rand.Float64() * zebras[i][d]
				zebras[i][d] += ps * rand.Float64() * mz
			}

			for d := range zebras[i] {
				zebras[i][d] = clamp(zebras[i][d], zebraRangeLow, zebraRangeHigh)
			}

			values[i] = fn.Evaluate(zebras[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), zebras[i]...)
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
		Method:     "zebra",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
