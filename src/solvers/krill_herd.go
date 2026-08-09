package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	krillHerdPopulation = 20
	krillHerdMaxSteps   = 1000
	krillHerdTolerance  = 1e-6
	krillHerdRangeLow   = -5.0
	krillHerdRangeHigh  = 5.0
)

// KrillHerd minimizuje konfigurisanu benchmark funkciju koristeći Krill Herd Algorithm: svaki krill se kreće pod
// uticajem indukovanog kretanja ka najboljem rešenju, skaliranog relativnom razlikom u vrednosti funkcije u odnosu na
// najgore rešenje u jatu
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func KrillHerd(problem models.Problem) (result models.Result, err error) {

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

	population := krillHerdPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := krillHerdMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := krillHerdTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	krills := make([][]float64, population)
	velocity := make([][]float64, population)
	values := make([]float64, population)
	for i := range krills {
		krill := make([]float64, dimensions)
		for d := range krill {
			krill[d] = krillHerdRangeLow + rand.Float64()*(krillHerdRangeHigh-krillHerdRangeLow)
		}
		krills[i] = krill
		velocity[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(krill)
	}

	best := append([]float64(nil), krills[0]...)
	bestValue := values[0]
	worstValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), krills[i]...)
		}
		if v > worstValue {
			worstValue = v
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// N_max=0.01 i faktor primene brzine 0.1 (efektivno ~0.001 preostale udaljenosti po
		// koraku) su previše konzervativni — konvergencija na sphere je stabilna ali toliko spora
		// da 1000 koraka nije dovoljno da stigne blizu 0. Pojačano na N_max=0.1 i faktor 0.3
		// (~0.03 preostale udaljenosti po koraku); provereno da ostaje stabilno bez divergencije
		// i na rastrigin/rosenbrock/ackley.
		nMax := 0.1
		omega := 0.9 * (1 - float64(steps)/float64(maxSteps))
		fitnessRange := worstValue - bestValue + 1e-10

		for i := range krills {
			attracted := (values[i] - bestValue) / fitnessRange

			for d := range krills[i] {
				velocity[i][d] = omega*velocity[i][d] + nMax*(1-attracted)*(best[d]-krills[i][d])
				krills[i][d] = krills[i][d] + 0.3*velocity[i][d]
				krills[i][d] = clamp(krills[i][d], krillHerdRangeLow, krillHerdRangeHigh)
			}

			values[i] = fn.Evaluate(krills[i])
		}

		bestValue = values[0]
		best = append([]float64(nil), krills[0]...)
		worstValue = values[0]
		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), krills[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
			if v > worstValue {
				worstValue = v
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
		Method:     "krill_herd",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
