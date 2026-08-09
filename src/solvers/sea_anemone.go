package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	seaAnemonePopulation = 20
	seaAnemoneMaxSteps   = 1000
	seaAnemoneTolerance  = 1e-6
	seaAnemoneRangeLow   = -5.0
	seaAnemoneRangeHigh  = 5.0
)

// SeaAnemone minimizuje konfigurisanu benchmark funkciju koristeći Sea Anemone Optimization: sasa kombinuje
// njihanje pipaka (nasumična perturbacija amplitudom koja opada tokom izvršavanja) sa simbiotskim privlačenjem
// ka najboljem rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SeaAnemone(problem models.Problem) (result models.Result, err error) {

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

	population := seaAnemonePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := seaAnemoneMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := seaAnemoneTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	anemones := make([][]float64, population)
	values := make([]float64, population)
	for i := range anemones {
		anemone := make([]float64, dimensions)
		for d := range anemone {
			anemone[d] = seaAnemoneRangeLow + rand.Float64()*(seaAnemoneRangeHigh-seaAnemoneRangeLow)
		}
		anemones[i] = anemone
		values[i] = fn.Evaluate(anemone)
	}

	best := append([]float64(nil), anemones[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), anemones[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range anemones {
			for d := range anemones[i] {
				sway := (rand.Float64()*2 - 1) * 0.4 * (1 - float64(steps)/float64(maxSteps))
				anemones[i][d] = anemones[i][d] + sway + rand.Float64()*(best[d]-anemones[i][d])*0.5
				anemones[i][d] = clamp(anemones[i][d], seaAnemoneRangeLow, seaAnemoneRangeHigh)
			}
			values[i] = fn.Evaluate(anemones[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), anemones[i]...)
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
		Method:     "sea_anemone",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
