package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	electricEelPopulation = 20
	electricEelMaxSteps   = 1000
	electricEelTolerance  = 1e-6
	electricEelRangeLow   = -5.0
	electricEelRangeHigh  = 5.0
)

// ElectricEel minimizuje konfigurisanu benchmark funkciju koristeći Electric Eel Foraging Optimization: jegulja
// nasumično bira interakciju (ka najboljem i nasumičnom peer-u), odmor (mala lokalna pretraga) ili lov sa
// pražnjenjem (jak impuls direktno ka najboljem)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ElectricEel(problem models.Problem) (result models.Result, err error) {

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

	population := electricEelPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := electricEelMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := electricEelTolerance
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

	eels := make([][]float64, population)
	values := make([]float64, population)
	for i := range eels {
		eel := make([]float64, dimensions)
		for d := range eel {
			eel[d] = electricEelRangeLow + rand.Float64()*(electricEelRangeHigh-electricEelRangeLow)
		}
		eels[i] = eel
		values[i] = fn.Evaluate(eel)
	}

	best := append([]float64(nil), eels[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), eels[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range eels {
			p := rand.Float64()
			switch {
			case p < 0.4:
				r := eels[randomIndexExcept(i)]
				for d := range eels[i] {
					eels[i][d] = eels[i][d] + rand.Float64()*(best[d]-eels[i][d]) + rand.Float64()*(r[d]-eels[i][d])*0.3
				}
			case p < 0.7:
				for d := range eels[i] {
					eels[i][d] = eels[i][d] + (rand.Float64()*2-1)*0.1
				}
			default:
				for d := range eels[i] {
					eels[i][d] = eels[i][d] + 0.8*(best[d]-eels[i][d])
				}
			}

			for d := range eels[i] {
				eels[i][d] = clamp(eels[i][d], electricEelRangeLow, electricEelRangeHigh)
			}
			values[i] = fn.Evaluate(eels[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), eels[i]...)
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
		Method:     "electric_eel",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
