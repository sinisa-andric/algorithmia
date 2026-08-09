package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	flowerPollinationFlowers   = 20
	flowerPollinationP         = 0.8
	flowerPollinationMaxSteps  = 500
	flowerPollinationTolerance = 1e-6
	flowerPollinationSpread    = 10.0
	flowerPollinationLevyScale = 0.01
	flowerPollinationLevyPower = 1.5
	flowerPollinationBound     = 15.0
)

// FlowerPollination minimizuje sphere funkciju koristeći Flower Pollination Algorithm: sa verovatnoćom p cvet
// izvršava globalnu (Levy-flight) polinaciju ka trenutno najboljem, inače lokalnu polinaciju između dva nasumična
// cveta
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func FlowerPollination(problem models.Problem) (result models.Result, err error) {

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

	flowerCount := flowerPollinationFlowers
	if v, ok := problem.Payload["flowers"].(float64); ok {
		flowerCount = int(v)
	}

	p := flowerPollinationP
	if v, ok := problem.Payload["p"].(float64); ok {
		p = v
	}

	maxSteps := flowerPollinationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := flowerPollinationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	flowers := make([][]float64, flowerCount)
	for i := range flowers {
		flower := make([]float64, dimensions)
		for d := range flower {
			flower[d] = problem.Point[d] + (rand.Float64()*2-1)*flowerPollinationSpread
		}
		flowers[i] = flower
	}

	best := append([]float64(nil), flowers[0]...)
	bestValue := fn.Evaluate(best)
	for _, flower := range flowers {
		if value := fn.Evaluate(flower); value < bestValue {
			bestValue = value
			best = append([]float64(nil), flower...)
		}
	}

	globalBest := append([]float64(nil), best...)
	globalBestValue := bestValue

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		for i, flower := range flowers {
			newFlower := make([]float64, dimensions)

			if rand.Float64() < p {
				for d := range newFlower {
					l := flowerPollinationLevyScale * randNorm() / math.Pow(math.Abs(randNorm()), 1/flowerPollinationLevyPower)
					newFlower[d] = flower[d] + l*(best[d]-flower[d])
				}
			} else {
				j := rand.IntN(flowerCount)
				k := rand.IntN(flowerCount)
				epsilon := rand.Float64()
				for d := range newFlower {
					newFlower[d] = flower[d] + epsilon*(flowers[j][d]-flowers[k][d])
				}
			}

			for d := range newFlower {
				if newFlower[d] > flowerPollinationBound {
					newFlower[d] = flowerPollinationBound
				} else if newFlower[d] < -flowerPollinationBound {
					newFlower[d] = -flowerPollinationBound
				}
			}

			if fn.Evaluate(newFlower) < fn.Evaluate(flower) {
				flowers[i] = newFlower
			}
		}

		bestValue = fn.Evaluate(flowers[0])
		best = append([]float64(nil), flowers[0]...)
		for _, flower := range flowers {
			if value := fn.Evaluate(flower); value < bestValue {
				bestValue = value
				best = append([]float64(nil), flower...)
			}
		}

		if bestValue < globalBestValue {
			globalBestValue = bestValue
			globalBest = append([]float64(nil), best...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "flower_pollination",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
