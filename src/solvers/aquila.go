package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	aquilaCount     = 20
	aquilaMaxSteps  = 500
	aquilaTolerance = 1e-6
	aquilaSpread    = 10.0
	aquilaAdjust    = 0.1
	aquilaLevyPower = 1.5
	aquilaBound     = 15.0
)

// Aquila minimizuje sphere funkciju koristeći Aquila Optimizer: prva polovina izvršavanja istražuje (proširena
// eksploracija oko sredine roja, ili suženo pretraživanje Levy-flight-om),
// druga polovina eksploatiše (kretanje u odnosu na nasumičnog para ili malu perturbaciju oko najboljeg)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Aquila(problem models.Problem) (result models.Result, err error) {

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

	count := aquilaCount
	if v, ok := problem.Payload["aquilas"].(float64); ok {
		count = int(v)
	}
	if count < 2 {
		count = 2
	}

	maxSteps := aquilaMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := aquilaTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	population := make([][]float64, count)
	for i := range population {
		aquila := make([]float64, dimensions)
		for d := range aquila {
			aquila[d] = problem.Point[d] + (rand.Float64()*2-1)*aquilaSpread
		}
		population[i] = aquila
	}

	best := append([]float64(nil), population[0]...)
	bestValue := fn.Evaluate(best)
	for _, aquila := range population {
		if value := fn.Evaluate(aquila); value < bestValue {
			bestValue = value
			best = append([]float64(nil), aquila...)
		}
	}

	globalBest := append([]float64(nil), best...)
	globalBestValue := bestValue

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		tRatio := float64(steps) / float64(maxSteps)

		mean := make([]float64, dimensions)
		for _, aquila := range population {
			for d := range aquila {
				mean[d] += aquila[d] / float64(count)
			}
		}

		next := make([][]float64, count)
		for i, aquila := range population {
			newAquila := make([]float64, dimensions)

			if tRatio < 0.5 {
				if rand.Float64() < 0.5 {
					for d := range newAquila {
						newAquila[d] = best[d]*(1-tRatio) + mean[d]*tRatio
					}
				} else {
					r := rand.Float64()
					theta := 2 * math.Pi * rand.Float64()
					for d := range newAquila {
						levy := randNorm() / math.Pow(math.Abs(randNorm()), 1/aquilaLevyPower)
						newAquila[d] = best[d] + levy*math.Cos(theta)*r
					}
				}
			} else {
				if rand.Float64() < 0.5 {
					other := population[rand.IntN(count)]
					for d := range newAquila {
						newAquila[d] = best[d] + rand.Float64()*(other[d]-aquila[d])
					}
				} else {
					for d := range newAquila {
						newAquila[d] = best[d] + aquilaAdjust*(rand.Float64()*2-1)
					}
				}
			}

			for d := range newAquila {
				if newAquila[d] > aquilaBound {
					newAquila[d] = aquilaBound
				} else if newAquila[d] < -aquilaBound {
					newAquila[d] = -aquilaBound
				}
			}

			next[i] = newAquila
		}

		population = next

		for _, aquila := range population {
			if value := fn.Evaluate(aquila); value < bestValue {
				bestValue = value
				best = append([]float64(nil), aquila...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, false)
			}
		}

		if bestValue < globalBestValue {
			globalBestValue = bestValue
			globalBest = append([]float64(nil), best...)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "aquila",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
