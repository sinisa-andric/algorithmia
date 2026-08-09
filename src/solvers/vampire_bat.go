package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	vampireBatPopulation = 20
	vampireBatMaxSteps   = 1000
	vampireBatTolerance  = 1e-6
	vampireBatRangeLow   = -5.0
	vampireBatRangeHigh  = 5.0
)

// VampireBat minimizuje konfigurisanu benchmark funkciju koristeći Vampire Bat Optimization: šišmiš se kreće ka
// najboljem rešenju brzinom skaliranom nasumičnom frekvencijom eholokacije, a povremeno dodatno deli hranu sa
// uspešnijim nasumičnim peer-om (regurgitacija)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func VampireBat(problem models.Problem) (result models.Result, err error) {

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

	population := vampireBatPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := vampireBatMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := vampireBatTolerance
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

	bats := make([][]float64, population)
	values := make([]float64, population)
	for i := range bats {
		bat := make([]float64, dimensions)
		for d := range bat {
			bat[d] = vampireBatRangeLow + rand.Float64()*(vampireBatRangeHigh-vampireBatRangeLow)
		}
		bats[i] = bat
		values[i] = fn.Evaluate(bat)
	}

	best := append([]float64(nil), bats[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), bats[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range bats {
			freq := rand.Float64()
			for d := range bats[i] {
				bats[i][d] = bats[i][d] + freq*(best[d]-bats[i][d])
			}

			if rand.Float64() < 0.2 {
				r := randomIndexExcept(i)
				if values[r] < values[i] {
					for d := range bats[i] {
						bats[i][d] = bats[i][d] + rand.Float64()*(bats[r][d]-bats[i][d])*0.3
					}
				}
			}

			// Bazno kretanje je isključivo ka best-u (uz povremeno pojačanje ka boljem peer-u), bez
			// ikakvog šuma — jato se brzo konsoliduje pre nego što stigne blizu optimuma. Povremena
			// opadajuća perturbacija održava istraživanje dovoljno dugo da se to izbegne
			if rand.Float64() < 0.2 {
				for d := range bats[i] {
					bats[i][d] = bats[i][d] + (rand.Float64()*2-1)*0.5*(1-float64(steps)/float64(maxSteps))
				}
			}

			for d := range bats[i] {
				bats[i][d] = clamp(bats[i][d], vampireBatRangeLow, vampireBatRangeHigh)
			}
			values[i] = fn.Evaluate(bats[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), bats[i]...)
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
		Method:     "vampire_bat",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
