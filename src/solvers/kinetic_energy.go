package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	kineticEnergyPopulation = 20
	kineticEnergyMaxSteps   = 500
	kineticEnergyTolerance  = 1e-6
	kineticEnergyRangeLow   = -5.0
	kineticEnergyRangeHigh  = 5.0
)

// KineticEnergy minimizuje konfigurisanu benchmark funkciju koristeći Kinetic Energy Optimization: svaka čestica
// nosi kinetičku energiju proporcionalnu udaljenosti od best-a koja pokreće njenu brzinu, uz povremene elastične
// sudare koji redistribuiraju brzinu i nezavisan šum koji sprečava potpuno zaustavljanje blizu best-a
// problem.Point inicijalizuje populaciju čestica i određuje njenu dimenzionalnost
func KineticEnergy(problem models.Problem) (result models.Result, err error) {

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

	population := kineticEnergyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := kineticEnergyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := kineticEnergyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	velocity := make([][]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = kineticEnergyRangeLow + rand.Float64()*(kineticEnergyRangeHigh-kineticEnergyRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
		velocity[i] = make([]float64, dimensions)
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
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = individuals[i][d] - best[d]
			}
			distance := norm(diff)
			ke := 0.5 * distance * distance

			for d := range individuals[i] {
				direction := 1.0
				if best[d] < individuals[i][d] {
					direction = -1.0
				} else if best[d] == individuals[i][d] {
					direction = 0.0
				}
				noise := (rand.Float64()*2 - 1) * 0.02
				velocity[i][d] = 0.7*velocity[i][d] + rand.Float64()*math.Sqrt(2*ke/float64(population))*direction*0.3 + noise

				if rand.Float64() < 0.1 {
					velocity[i][d] *= -0.5
				}

				individuals[i][d] = clamp(individuals[i][d]+velocity[i][d], kineticEnergyRangeLow, kineticEnergyRangeHigh)
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
		Method:     "kinetic_energy",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
