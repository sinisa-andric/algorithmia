package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	studentPsychologyPopulation = 20
	studentPsychologyMaxSteps   = 500
	studentPsychologyTolerance  = 1e-6
	studentPsychologyRangeLow   = -5.0
	studentPsychologyRangeHigh  = 5.0
)

// StudentPsychology minimizuje konfigurisanu benchmark funkciju koristeći Student Psychology Based Optimization:
// najbolji student se samostalno usavršava nezavisnom perturbacijom, dok ostali nasumično prate ili najboljeg
// studenta ili prosek razreda
// problem.Point inicijalizuje populaciju studenata i određuje njenu dimenzionalnost
func StudentPsychology(problem models.Problem) (result models.Result, err error) {

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

	population := studentPsychologyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := studentPsychologyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := studentPsychologyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = studentPsychologyRangeLow + rand.Float64()*(studentPsychologyRangeHigh-studentPsychologyRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		mean := make([]float64, dimensions)
		for _, individual := range individuals {
			for d := range mean {
				mean[d] += individual[d]
			}
		}
		for d := range mean {
			mean[d] /= float64(len(individuals))
		}

		bestIdx := 0
		for i, v := range values {
			if v < values[bestIdx] {
				bestIdx = i
			}
		}

		for i := range individuals {
			switch {
			case i == bestIdx:
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.2*(1-float64(steps)/float64(maxSteps)), studentPsychologyRangeLow, studentPsychologyRangeHigh)
				}
			case rand.Float64() < 0.5:
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*(best[d]-individuals[i][d])*0.5, studentPsychologyRangeLow, studentPsychologyRangeHigh)
				}
			default:
				for d := range individuals[i] {
					pull := rand.Float64() * (mean[d] - individuals[i][d]) * 0.4
					noise := (rand.Float64()*2 - 1) * 0.2
					individuals[i][d] = clamp(individuals[i][d]+pull+noise, studentPsychologyRangeLow, studentPsychologyRangeHigh)
				}
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "student_psychology",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
