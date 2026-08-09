package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mosquitoPopulation = 20
	mosquitoMaxSteps   = 1000
	mosquitoTolerance  = 1e-6
	mosquitoRangeLow   = -5.0
	mosquitoRangeHigh  = 5.0
)

// Mosquito minimizuje konfigurisanu benchmark funkciju koristeći Mosquito Host-Seeking Optimization: komarac prati
// CO2 trag ka domaćinu (najboljem rešenju) jačinom proporcionalnom snazi traga,
// ili gubi trag i nasumično leti u istraživanje prostora
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Mosquito(problem models.Problem) (result models.Result, err error) {

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

	population := mosquitoPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mosquitoMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mosquitoTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	mosquitoes := make([][]float64, population)
	values := make([]float64, population)
	for i := range mosquitoes {
		mosquito := make([]float64, dimensions)
		for d := range mosquito {
			mosquito[d] = mosquitoRangeLow + rand.Float64()*(mosquitoRangeHigh-mosquitoRangeLow)
		}
		mosquitoes[i] = mosquito
		values[i] = fn.Evaluate(mosquito)
	}

	best := append([]float64(nil), mosquitoes[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), mosquitoes[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range mosquitoes {
			plumeStrength := 1 / (values[i] + 1e-10)

			if rand.Float64() < 0.7 {
				factor := plumeStrength / (plumeStrength + 1)
				for d := range mosquitoes[i] {
					mosquitoes[i][d] = mosquitoes[i][d] + rand.Float64()*(best[d]-mosquitoes[i][d])*factor
				}
			} else {
				for d := range mosquitoes[i] {
					mosquitoes[i][d] = mosquitoes[i][d] + (rand.Float64()*2-1)*1.0
				}
			}

			for d := range mosquitoes[i] {
				mosquitoes[i][d] = clamp(mosquitoes[i][d], mosquitoRangeLow, mosquitoRangeHigh)
			}
			values[i] = fn.Evaluate(mosquitoes[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), mosquitoes[i]...)
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
		Method:     "mosquito",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
