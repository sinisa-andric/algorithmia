package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	birdsOfParadisePopulation = 20
	birdsOfParadiseMaxSteps   = 1000
	birdsOfParadiseTolerance  = 1e-6
	birdsOfParadiseAlpha      = 0.5
	birdsOfParadiseRange      = 5.0
	birdsOfParadiseLevyLambda = 1.5
)

// BirdsOfParadise minimizuje konfigurisanu benchmark funkciju koristeći algoritam ptice raja: svaka ptica naizmenično
// vrši display (eksploracija — Levy-flight ka boljoj ptici ili nasumična perturbacija) i feeding (eksploatacija —
// kretanje ka globalnom najboljem skaliran razlikom fitnes vrednosti)
// problem.Point inicijalizuje jato i određuje njenu dimenzionalnost
func BirdsOfParadise(problem models.Problem) (result models.Result, err error) {

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

	population := birdsOfParadisePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := birdsOfParadiseMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := birdsOfParadiseTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	alpha := birdsOfParadiseAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	positions := make([][]float64, population)
	values := make([]float64, population)
	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = (rand.Float64()*2 - 1) * birdsOfParadiseRange
		}
		positions[i] = position
		values[i] = fn.Evaluate(position)
	}

	best := append([]float64(nil), positions[0]...)
	bestValue := values[0]
	for i, val := range values {
		if val < bestValue {
			bestValue = val
			best = append([]float64(nil), positions[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)

	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := 0; i < population; i++ {
			newPos := make([]float64, dimensions)

			if rand.Float64() < alpha {
				j := rand.IntN(population - 1)
				if j >= i {
					j++
				}

				if values[j] < values[i] {
					levyStep := mantegnaLevy(birdsOfParadiseLevyLambda)
					for d := range newPos {
						newPos[d] = positions[i][d] + levyStep*(positions[j][d]-positions[i][d])
					}
				} else {
					for d := range newPos {
						perturbScale := 0.1*math.Abs(positions[i][d]) + 0.01
						newPos[d] = positions[i][d] + (rand.Float64()-0.5)*perturbScale
					}
				}
			} else {
				diff := math.Abs(values[i]-bestValue) / (math.Abs(bestValue) + 1e-10)
				for d := range newPos {
					newPos[d] = best[d] + (rand.Float64()*2-1)*diff*(positions[i][d]-best[d])
				}
			}

			newValue := fn.Evaluate(newPos)
			if newValue < values[i] {
				positions[i] = newPos
				values[i] = newValue
				if newValue < bestValue {
					bestValue = newValue
					best = append([]float64(nil), newPos...)
				}
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "birds_of_paradise",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
