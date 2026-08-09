package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	goldenEaglePopulation = 20
	goldenEagleMaxSteps   = 1000
	goldenEagleTolerance  = 1e-6
	goldenEagleRangeLow   = -5.0
	goldenEagleRangeHigh  = 5.0
)

// GoldenEagle minimizuje konfigurisanu benchmark funkciju koristeći Golden Eagle Optimizer: brzina svakog orla spaja
// normalizovan smer ka plenu (najboljem rešenju) i nasumičan smer kruženja,
// ponderisane sklonošću ka napadu koja raste dok sklonost ka kruženju opada tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GoldenEagle(problem models.Problem) (result models.Result, err error) {

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

	population := goldenEaglePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := goldenEagleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := goldenEagleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	eagles := make([][]float64, population)
	velocity := make([][]float64, population)
	values := make([]float64, population)
	for i := range eagles {
		eagle := make([]float64, dimensions)
		for d := range eagle {
			eagle[d] = goldenEagleRangeLow + rand.Float64()*(goldenEagleRangeHigh-goldenEagleRangeLow)
		}
		eagles[i] = eagle
		velocity[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(eagle)
	}

	best := append([]float64(nil), eagles[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), eagles[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		attackTendency := 0.5 - float64(steps)/(2*float64(maxSteps))
		cruiseTendency := 1 - attackTendency

		for i := range eagles {
			preyDir := make([]float64, dimensions)
			for d := range preyDir {
				preyDir[d] = best[d] - eagles[i][d]
			}
			if n := norm(preyDir); n > 1e-10 {
				for d := range preyDir {
					preyDir[d] /= n
				}
			}

			randDir := make([]float64, dimensions)
			for d := range randDir {
				randDir[d] = rand.Float64()*2 - 1
			}
			if n := norm(randDir); n > 1e-10 {
				for d := range randDir {
					randDir[d] /= n
				}
			}

			for d := range eagles[i] {
				velocity[i][d] = rand.Float64()*(attackTendency*preyDir[d]+cruiseTendency*randDir[d]) + rand.Float64()*velocity[i][d]
				eagles[i][d] += velocity[i][d]
				eagles[i][d] = clamp(eagles[i][d], goldenEagleRangeLow, goldenEagleRangeHigh)
			}

			values[i] = fn.Evaluate(eagles[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), eagles[i]...)
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
		Method:     "golden_eagle",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
