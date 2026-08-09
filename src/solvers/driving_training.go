package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	drivingTrainingPopulation = 20
	drivingTrainingMaxSteps   = 500
	drivingTrainingTolerance  = 1e-6
	drivingTrainingRangeLow   = -5.0
	drivingTrainingRangeHigh  = 5.0
)

// DrivingTraining minimizuje konfigurisanu benchmark funkciju koristeći Driving Training-Based Optimization:
// kombinuje fazu podučavanja (pomak ka instruktoru/best-u) i fazu samostalne vožnje (nezavisna perturbacija uz
// blagu vezu ka best-u) u jedan predlog koji se pohlepno prihvata samo ako poboljšava rezultat
// problem.Point inicijalizuje populaciju vozača i određuje njenu dimenzionalnost
func DrivingTraining(problem models.Problem) (result models.Result, err error) {

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

	population := drivingTrainingPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := drivingTrainingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := drivingTrainingTolerance
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
			individual[d] = drivingTrainingRangeLow + rand.Float64()*(drivingTrainingRangeHigh-drivingTrainingRangeLow)
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

		for i := range individuals {
			candidate := make([]float64, dimensions)
			for d := range candidate {
				teach := rand.Float64() * (best[d] - individuals[i][d]) * 0.5
				selfDrive := (rand.Float64()*2-1)*0.4*(1-float64(steps)/float64(maxSteps)) + rand.Float64()*(best[d]-individuals[i][d])*0.2
				candidate[d] = clamp(individuals[i][d]+teach+selfDrive, drivingTrainingRangeLow, drivingTrainingRangeHigh)
			}
			candidateValue := fn.Evaluate(candidate)
			if candidateValue < values[i] {
				individuals[i] = candidate
				values[i] = candidateValue
			}
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
		Method:     "driving_training",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
