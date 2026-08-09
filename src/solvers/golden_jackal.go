package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	goldenJackalPopulation = 20
	goldenJackalMaxSteps   = 1000
	goldenJackalTolerance  = 1e-6
	goldenJackalRangeLow   = -5.0
	goldenJackalRangeHigh  = 5.0
)

// GoldenJackal minimizuje konfigurisanu benchmark funkciju koristeći Golden Jackal Optimization: mužjak (najbolje
// rešenje) i ženka (drugo najbolje) vode potragu za plenom, a svaki šakal se pomera na sredinu između svog pristupa
// mužjaku i ženci, ponderisanog energijom koja opada tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GoldenJackal(problem models.Problem) (result models.Result, err error) {

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

	population := goldenJackalPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := goldenJackalMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := goldenJackalTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	jackals := make([][]float64, population)
	values := make([]float64, population)
	for i := range jackals {
		jackal := make([]float64, dimensions)
		for d := range jackal {
			jackal[d] = goldenJackalRangeLow + rand.Float64()*(goldenJackalRangeHigh-goldenJackalRangeLow)
		}
		jackals[i] = jackal
		values[i] = fn.Evaluate(jackal)
	}

	best, secondBest := topTwo(jackals, values)
	bestValue := fn.Evaluate(best)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// e1 je opadajuća amplituda energije (1.5 -> 0) koja kontroliše koliko daleko
		// agent sme da se pomeri u odnosu na male/female u ovom koraku.
		e1 := 1.5 * (1 - float64(steps)/float64(maxSteps))

		for i := range jackals {
			// Svaki agent u ovom koraku prati ili male ili female (50/50 šansa).
			if rand.Float64() < 0.5 {
				for d := range jackals[i] {
					// D_male je nepredznačna udaljenost od male po ovoj dimenziji.
					dMale := math.Abs(best[d] - jackals[i][d])
					// e0 je predznačena "energija bekstva plena" u [-1, 1]: bez ovog
					// predznaka bi e1*rand*dMale bio uvek >= 0, pa bi jackals[i][d]
					// uvek završavao <= best[d] — sistematski drift ka donjoj granici
					// opsega umesto konvergencije ka pravom optimumu.
					e0 := rand.Float64()*2 - 1
					jackals[i][d] = best[d] - e1*e0*dMale
				}
			} else {
				for d := range jackals[i] {
					dFemale := math.Abs(secondBest[d] - jackals[i][d])
					e0 := rand.Float64()*2 - 1
					jackals[i][d] = secondBest[d] - e1*e0*dFemale
				}
			}
			for d := range jackals[i] {
				jackals[i][d] = clamp(jackals[i][d], goldenJackalRangeLow, goldenJackalRangeHigh)
			}
			values[i] = fn.Evaluate(jackals[i])
		}

		// male/female se ažuriraju na kraju koraka na osnovu novih pozicija/vrednosti,
		// pa sledeći korak uvek cilja ka trenutno najbolje pronađenim rešenjima.

		best, secondBest = topTwo(jackals, values)
		bestValue = fn.Evaluate(best)

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
		Method:     "golden_jackal",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// topTwo vraća kopije dve pozicije sa najmanjim vrednostima iz agents/values
func topTwo(agents [][]float64, values []float64) (first, second []float64) {

	bestIdx, secondIdx := 0, 1
	if values[secondIdx] < values[bestIdx] {
		bestIdx, secondIdx = secondIdx, bestIdx
	}

	for i := 2; i < len(agents); i++ {
		switch {
		case values[i] < values[bestIdx]:
			secondIdx = bestIdx
			bestIdx = i
		case values[i] < values[secondIdx]:
			secondIdx = i
		}
	}

	return append([]float64(nil), agents[bestIdx]...), append([]float64(nil), agents[secondIdx]...)
}
