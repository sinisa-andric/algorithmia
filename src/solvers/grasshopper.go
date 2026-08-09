package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	grasshopperCount     = 20
	grasshopperMaxSteps  = 500
	grasshopperTolerance = 1e-6
	grasshopperSpread    = 10.0
	grasshopperCMax      = 1.0
	grasshopperCMin      = 0.00001
	grasshopperSocialA   = 0.5
	grasshopperSocialB   = 1.5
	grasshopperBound     = 15.0
	grasshopperAttract   = 0.1
)

// Grasshopper minimizuje sphere funkciju koristeći Grasshopper Optimization Algorithm: svaki skakavac se premešta
// socijalnim (privlačenje/odbijanje) silama od svakog drugog skakavca plus privlačenjem ka trenutno najboljem,
// pri čemu socijalni koeficijent c linearno opada tokom izvršavanja
// problem.Point inicijalizuje roj i određuje njegovu dimenzionalnost
func Grasshopper(problem models.Problem) (result models.Result, err error) {

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

	count := grasshopperCount
	if v, ok := problem.Payload["grasshoppers"].(float64); ok {
		count = int(v)
	}

	maxSteps := grasshopperMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := grasshopperTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	swarm := make([][]float64, count)
	for i := range swarm {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = problem.Point[d] + (rand.Float64()*2-1)*grasshopperSpread
		}
		swarm[i] = position
	}

	best := append([]float64(nil), swarm[0]...)
	bestValue := fn.Evaluate(best)
	for _, position := range swarm {
		if value := fn.Evaluate(position); value < bestValue {
			bestValue = value
			best = append([]float64(nil), position...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		c := grasshopperCMax - float64(steps)*(grasshopperCMax-grasshopperCMin)/float64(maxSteps)

		next := make([][]float64, count)
		for i := range swarm {
			sum := make([]float64, dimensions)

			for j := range swarm {
				if j == i {
					continue
				}

				distance := 0.0
				for d := 0; d < dimensions; d++ {
					diff := swarm[j][d] - swarm[i][d]
					distance += diff * diff
				}
				distance = math.Sqrt(distance)
				if distance == 0 {
					continue
				}

				social := grasshopperSocialA*math.Exp(-distance/grasshopperSocialB) - math.Exp(-distance)

				for d := 0; d < dimensions; d++ {
					sum[d] += c * social * (swarm[j][d] - swarm[i][d]) / distance
				}
			}

			position := make([]float64, dimensions)
			for d := range position {
				position[d] = sum[d] + best[d]
				position[d] += grasshopperAttract * (best[d] - position[d])

				if position[d] > grasshopperBound {
					position[d] = grasshopperBound
				} else if position[d] < -grasshopperBound {
					position[d] = -grasshopperBound
				}
			}
			next[i] = position
		}

		swarm = next

		for _, position := range swarm {
			if value := fn.Evaluate(position); value < bestValue {
				bestValue = value
				best = append([]float64(nil), position...)
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
		Method:     "grasshopper",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
