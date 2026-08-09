package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	marinePredatorsPopulation = 20
	marinePredatorsMaxSteps   = 1000
	marinePredatorsTolerance  = 1e-6
	marinePredatorsRangeLow   = -5.0
	marinePredatorsRangeHigh  = 5.0
	marinePredatorsLevyLambda = 1.5
)

// MarinePredators minimizuje konfigurisanu benchmark funkciju koristeći Marine Predators Algorithm:
// izvršavanje prolazi kroz tri faze sa različitim odnosom brzine plena i predatora: Levy-flight ka najboljem,
// mešoviti režim (deo linearno prati najboljeg, deo koristi Levy-flight), i čisto Levy-flight sa opadajućim faktorom
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MarinePredators(problem models.Problem) (result models.Result, err error) {

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

	population := marinePredatorsPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := marinePredatorsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := marinePredatorsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	prey := make([][]float64, population)
	values := make([]float64, population)
	for i := range prey {
		p := make([]float64, dimensions)
		for d := range p {
			p[d] = marinePredatorsRangeLow + rand.Float64()*(marinePredatorsRangeHigh-marinePredatorsRangeLow)
		}
		prey[i] = p
		values[i] = fn.Evaluate(p)
	}

	best := append([]float64(nil), prey[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), prey[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		fraction := float64(steps) / float64(maxSteps)
		cf := math.Pow(1-fraction, 2*fraction)

		for i := range prey {
			switch {
			case float64(steps) < float64(maxSteps)/3:
				levy := mantegnaLevy(marinePredatorsLevyLambda)
				for d := range prey[i] {
					prey[i][d] = prey[i][d] + levy*rand.Float64()*(best[d]-prey[i][d])
				}
			case float64(steps) < 2*float64(maxSteps)/3:
				if i < population/2 {
					for d := range prey[i] {
						prey[i][d] += rand.Float64() * (best[d] - prey[i][d])
					}
				} else {
					levy := mantegnaLevy(marinePredatorsLevyLambda)
					for d := range prey[i] {
						prey[i][d] = best[d] + cf*levy*(best[d]-prey[i][d])
					}
				}
			default:
				levy := mantegnaLevy(marinePredatorsLevyLambda)
				for d := range prey[i] {
					prey[i][d] = best[d] + cf*levy*(best[d]-prey[i][d])
				}
			}

			for d := range prey[i] {
				prey[i][d] = clamp(prey[i][d], marinePredatorsRangeLow, marinePredatorsRangeHigh)
			}

			values[i] = fn.Evaluate(prey[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), prey[i]...)
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
		Method:     "marine_predators",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
