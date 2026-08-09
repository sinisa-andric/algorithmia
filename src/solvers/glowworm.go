package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	glowwormPopulation = 20
	glowwormMaxSteps   = 1000
	glowwormTolerance  = 1e-6
	glowwormRangeLow   = -5.0
	glowwormRangeHigh  = 5.0
)

// Glowworm minimizuje konfigurisanu benchmark funkciju koristeći Glowworm Swarm Optimization: svaki svitac emituje
// lucifern obrnuto proporcionalan vrednosti funkcije, zatim se kreće ka susedu unutar radijusa vidljivosti koji ima
// veći intenzitet luciferina
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Glowworm(problem models.Problem) (result models.Result, err error) {

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

	population := glowwormPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := glowwormMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := glowwormTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	glowworms := make([][]float64, population)
	values := make([]float64, population)
	luciferin := make([]float64, population)
	for i := range glowworms {
		glowworm := make([]float64, dimensions)
		for d := range glowworm {
			glowworm[d] = glowwormRangeLow + rand.Float64()*(glowwormRangeHigh-glowwormRangeLow)
		}
		glowworms[i] = glowworm
		values[i] = fn.Evaluate(glowworm)
	}

	best := append([]float64(nil), glowworms[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), glowworms[i]...)
		}
	}

	const neighborhoodRadius = 2.0
	const stepSize = 0.03

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range glowworms {
			luciferin[i] = 0.4*luciferin[i] + 0.6*(1/(values[i]+1e-10))
		}

		for i := range glowworms {
			neighbors := make([]int, 0)
			for j := range glowworms {
				if j == i {
					continue
				}
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = glowworms[i][d] - glowworms[j][d]
				}
				if norm(diff) < neighborhoodRadius && luciferin[j] > luciferin[i] {
					neighbors = append(neighbors, j)
				}
			}

			if len(neighbors) == 0 {
				for d := range glowworms[i] {
					glowworms[i][d] = glowworms[i][d] + (rand.Float64()*2-1)*0.1
				}
			} else {
				totalLuciferin := 0.0
				for _, j := range neighbors {
					totalLuciferin += luciferin[j]
				}
				pick := rand.Float64() * totalLuciferin
				chosen := neighbors[0]
				cumulative := 0.0
				for _, j := range neighbors {
					cumulative += luciferin[j]
					if cumulative >= pick {
						chosen = j
						break
					}
				}

				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = glowworms[chosen][d] - glowworms[i][d]
				}
				normDiff := norm(diff) + 1e-10
				for d := range glowworms[i] {
					glowworms[i][d] = glowworms[i][d] + stepSize*diff[d]/normDiff
				}
			}

			for d := range glowworms[i] {
				glowworms[i][d] = clamp(glowworms[i][d], glowwormRangeLow, glowwormRangeHigh)
			}
			values[i] = fn.Evaluate(glowworms[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), glowworms[i]...)
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
		Method:     "glowworm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
