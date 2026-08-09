package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	acoAbcPopulation  = 20
	acoAbcMaxSteps    = 500
	acoAbcEvaporation = 0.5
	acoAbcLimit       = 50
	acoAbcTolerance   = 1e-6
	acoAbcRangeLow    = -5.0
	acoAbcRangeHigh   = 5.0
)

// AcoAbc minimizuje konfigurisanu benchmark funkciju koristeći ACO-ABC hibrid: u ACO fazi svako rešenje uzorkuje
// novu tačku iz Gausove raspodele čija disperzija zavisi od rastojanja do nasumičnog peer-a, u ABC fazi se rešenja
// biraju ruletom srazmerno fitnesu i pomeraju ka najboljem, a rešenja koja dugo ne napreduju se nasumično resetuju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func AcoAbc(problem models.Problem) (result models.Result, err error) {

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

	population := acoAbcPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := acoAbcMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	evaporation := acoAbcEvaporation
	if v, ok := problem.Payload["evaporation"].(float64); ok {
		evaporation = v
	}

	limit := acoAbcLimit
	if v, ok := problem.Payload["limit"].(float64); ok {
		limit = int(v)
	}

	tolerance := acoAbcTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	randomIndexExcept := func(exclude int) int {
		if population < 2 {
			return exclude
		}
		j := rand.IntN(population)
		for j == exclude {
			j = rand.IntN(population)
		}
		return j
	}

	solutions := make([][]float64, population)
	values := make([]float64, population)
	trialCount := make([]int, population)
	for i := range solutions {
		solution := make([]float64, dimensions)
		for d := range solution {
			solution[d] = acoAbcRangeLow + rand.Float64()*(acoAbcRangeHigh-acoAbcRangeLow)
		}
		solutions[i] = solution
		values[i] = fn.Evaluate(solution)
	}

	best := append([]float64(nil), solutions[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), solutions[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// ACO faza (employed bees analog) — difuzija oko svakog rešenja skalirana rastojanjem do
		// nasumičnog peer-a; ovo je nezavisan mehanizam od ABC faze ispod (Gausovo uzorkovanje, a ne
		// selekcija i usmereno kretanje ka best-u)
		for i := range solutions {
			r := randomIndexExcept(i)
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = solutions[i][d] - solutions[r][d]
			}
			sigma := evaporation * norm(diff)

			candidate := make([]float64, dimensions)
			for d := range candidate {
				candidate[d] = clamp(solutions[i][d]+randNorm()*sigma, acoAbcRangeLow, acoAbcRangeHigh)
			}
			candidateValue := fn.Evaluate(candidate)

			if candidateValue < values[i] {
				solutions[i] = candidate
				values[i] = candidateValue
				trialCount[i] = 0
			} else {
				trialCount[i]++
			}
		}

		// ABC faza (onlooker bees analog) — ruletom izabrana rešenja (proporcionalno fitnesu) se
		// usmereno pomeraju ka best-u, za razliku od nezavisnog Gausovog uzorkovanja u ACO fazi iznad
		fitness := make([]float64, population)
		totalFitness := 0.0
		for i, v := range values {
			if v >= 0 {
				fitness[i] = 1 / (1 + v)
			} else {
				fitness[i] = 1 + math.Abs(v)
			}
			totalFitness += fitness[i]
		}
		for range population {
			idx := population - 1
			if totalFitness > 0 {
				pick := rand.Float64() * totalFitness
				cumulative := 0.0
				for i, f := range fitness {
					cumulative += f
					if cumulative >= pick {
						idx = i
						break
					}
				}
			} else {
				idx = rand.IntN(population)
			}

			for d := range solutions[idx] {
				solutions[idx][d] = clamp(solutions[idx][d]+rand.Float64()*(best[d]-solutions[idx][d])*0.5, acoAbcRangeLow, acoAbcRangeHigh)
			}
			values[idx] = fn.Evaluate(solutions[idx])
			if values[idx] < bestValue {
				trialCount[idx] = 0
			}
		}

		// scout faza — rešenja koja dugo ne napreduju (ni ACO ni ABC faza ih nije poboljšala) se
		// nasumično resetuju radi diverziteta
		for i := range solutions {
			if trialCount[i] > limit {
				for d := range solutions[i] {
					solutions[i][d] = acoAbcRangeLow + rand.Float64()*(acoAbcRangeHigh-acoAbcRangeLow)
				}
				values[i] = fn.Evaluate(solutions[i])
				trialCount[i] = 0
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), solutions[i]...)
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
		Method:     "aco_abc",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
