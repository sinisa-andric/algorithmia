package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	cheetahPopulation = 20
	cheetahMaxSteps   = 1000
	cheetahTolerance  = 1e-6
	cheetahRangeLow   = -5.0
	cheetahRangeHigh  = 5.0
)

// Cheetah minimizuje konfigurisanu benchmark funkciju koristeći Cheetah Optimizer: svaki gepard nasumično bira jedan
// od četiri režima kretanja u odnosu na najbolje rešenje — sporo hodanje, trčanje,
// napad koji se prigušuje tokom izvršavanja, ili odmor uz Gausov šum
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Cheetah(problem models.Problem) (result models.Result, err error) {

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

	population := cheetahPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := cheetahMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := cheetahTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	cheetahs := make([][]float64, population)
	values := make([]float64, population)
	for i := range cheetahs {
		cheetah := make([]float64, dimensions)
		for d := range cheetah {
			cheetah[d] = cheetahRangeLow + rand.Float64()*(cheetahRangeHigh-cheetahRangeLow)
		}
		cheetahs[i] = cheetah
		values[i] = fn.Evaluate(cheetah)
	}

	best := append([]float64(nil), cheetahs[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), cheetahs[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range cheetahs {
			t := int(math.Round(1 + rand.Float64()*3))

			switch t {
			case 1:
				for d := range cheetahs[i] {
					cheetahs[i][d] += 0.25 * rand.Float64() * (best[d] - cheetahs[i][d])
				}
			case 2:
				for d := range cheetahs[i] {
					cheetahs[i][d] += rand.Float64() * (best[d] - cheetahs[i][d])
				}
			case 3:
				factor := math.Exp(-float64(steps) / float64(maxSteps))
				for d := range cheetahs[i] {
					cheetahs[i][d] = best[d] + rand.Float64()*(cheetahs[i][d]-best[d])*factor
				}
			default:
				for d := range cheetahs[i] {
					cheetahs[i][d] += 0.1 * randNorm()
				}
			}

			for d := range cheetahs[i] {
				cheetahs[i][d] = clamp(cheetahs[i][d], cheetahRangeLow, cheetahRangeHigh)
			}

			values[i] = fn.Evaluate(cheetahs[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), cheetahs[i]...)
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
		Method:     "cheetah",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
