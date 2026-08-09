package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	whaleOptimizationWhales    = 20
	whaleOptimizationMaxSteps  = 500
	whaleOptimizationTolerance = 1e-6
	whaleOptimizationSpread    = 10.0
	whaleOptimizationSpiralB   = 1.0
)

// WhaleOptimization minimizuje sphere funkciju koristeći Whale Optimization Algorithm: kitovi naizmenično
// opkoljavaju/traže plen i izvode spiralni napad mehurićima oko trenutno najboljeg
// problem.Point inicijalizuje jato i određuje njegovu dimenzionalnost
func WhaleOptimization(problem models.Problem) (result models.Result, err error) {

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

	whales := whaleOptimizationWhales
	if v, ok := problem.Payload["whales"].(float64); ok {
		whales = int(v)
	}

	maxSteps := whaleOptimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := whaleOptimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	pod := make([][]float64, whales)
	for i := range pod {
		whale := make([]float64, dimensions)
		for d := range whale {
			whale[d] = problem.Point[d] + (rand.Float64()*2-1)*whaleOptimizationSpread
		}
		pod[i] = whale
	}

	best := append([]float64(nil), pod[0]...)
	bestValue := fn.Evaluate(best)
	for _, whale := range pod {
		if value := fn.Evaluate(whale); value < bestValue {
			bestValue = value
			best = append([]float64(nil), whale...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		a := 2.0 - 2.0*float64(steps)/float64(maxSteps)

		for i, whale := range pod {
			A := a * (2*rand.Float64() - 1)
			c := 2 * rand.Float64()

			if rand.Float64() < 0.5 {
				if math.Abs(A) < 1 {
					for d := range whale {
						dist := math.Abs(c*best[d] - whale[d])
						whale[d] = best[d] - A*dist
					}
				} else {
					other := rand.IntN(whales)
					for other == i && whales > 1 {
						other = rand.IntN(whales)
					}
					randomWhale := pod[other]

					for d := range whale {
						dist := math.Abs(c*randomWhale[d] - whale[d])
						whale[d] = randomWhale[d] - A*dist
					}
				}
			} else {
				l := rand.Float64()*2 - 1
				for d := range whale {
					dist := math.Abs(best[d] - whale[d])
					whale[d] = dist*math.Exp(whaleOptimizationSpiralB*l)*math.Cos(2*math.Pi*l) + best[d]
				}
			}

			pod[i] = whale

			if value := fn.Evaluate(whale); value < bestValue {
				bestValue = value
				best = append([]float64(nil), whale...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "whale_optimization",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
