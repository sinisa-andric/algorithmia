package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	chimpPopulation = 20
	chimpMaxSteps   = 1000
	chimpTolerance  = 1e-6
	chimpRangeLow   = -5.0
	chimpRangeHigh  = 5.0
)

// Chimp minimizuje konfigurisanu benchmark funkciju koristeći Chimp Optimization Algorithm: četiri vodeće čimpanze
// (napadač, prepreka, gonič i vozač) vode ostatak čopora, svaka čimpanza se pomera na prosek svog pristupa sve četiri
// vodeće jedinke
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Chimp(problem models.Problem) (result models.Result, err error) {

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

	population := chimpPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := chimpMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := chimpTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sortByValue := func(points [][]float64, vals []float64) {
		idx := make([]int, len(points))
		for i := range idx {
			idx[i] = i
		}
		for i := 1; i < len(idx); i++ {
			for j := i; j > 0 && vals[idx[j]] < vals[idx[j-1]]; j-- {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			}
		}
		sortedPoints := make([][]float64, len(points))
		sortedVals := make([]float64, len(points))
		for i, k := range idx {
			sortedPoints[i] = points[k]
			sortedVals[i] = vals[k]
		}
		copy(points, sortedPoints)
		copy(vals, sortedVals)
	}

	chimps := make([][]float64, population)
	values := make([]float64, population)
	for i := range chimps {
		chimp := make([]float64, dimensions)
		for d := range chimp {
			chimp[d] = chimpRangeLow + rand.Float64()*(chimpRangeHigh-chimpRangeLow)
		}
		chimps[i] = chimp
		values[i] = fn.Evaluate(chimp)
	}
	sortByValue(chimps, values)

	leaderIdx := func(rank int) int {
		return min(rank, population-1)
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := values[0]

		attacker := chimps[leaderIdx(0)]
		barrier := chimps[leaderIdx(1)]
		chaser := chimps[leaderIdx(2)]
		driver := chimps[leaderIdx(3)]

		// Petlja mora da obuhvati i same lidere (indeksi 0..3), ne samo sledbenike — u suprotnom
		// lideri nikad ne dobijaju formulu kretanja, ostaju zamrznuti na poziciji iz trenutka kad
		// su prvi put postali najbolja četvorka, i ceo čopor je zauvek ograničen tom vrednošću.
		for i := range population {
			for d := range chimps[i] {
				// e1..e4 su predznačeni koeficijenti u [-1, 1]. Bez predznaka (čist rand u [0,1))
				// Xk = leader - rand*|...| uvek oduzima nenegativnu vrednost od leader-a, pa svaka
				// dimenzija sistematski klizi u jednom smeru dok se ne zaglavi na granici opsega —
				// isti obrazac greške kao ranije u golden_jackal.go.
				e1 := rand.Float64()*2 - 1
				e2 := rand.Float64()*2 - 1
				e3 := rand.Float64()*2 - 1
				e4 := rand.Float64()*2 - 1
				x1 := attacker[d] - e1*math.Abs(rand.Float64()*attacker[d]-chimps[i][d])
				x2 := barrier[d] - e2*math.Abs(rand.Float64()*barrier[d]-chimps[i][d])
				x3 := chaser[d] - e3*math.Abs(rand.Float64()*chaser[d]-chimps[i][d])
				x4 := driver[d] - e4*math.Abs(rand.Float64()*driver[d]-chimps[i][d])
				chimps[i][d] = (x1 + x2 + x3 + x4) / 4
				chimps[i][d] = clamp(chimps[i][d], chimpRangeLow, chimpRangeHigh)
			}
			values[i] = fn.Evaluate(chimps[i])
		}

		sortByValue(chimps, values)

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, chimps[0], values[0], false)
		}

		if math.Abs(values[0]-prevBestValue) < tolerance {
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
		trajectory = recordTrajectory(trajectory, steps, chimps[0], values[0], true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "chimp",
		Point:      chimps[0],
		Value:      values[0],
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
