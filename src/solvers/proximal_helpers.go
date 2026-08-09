package solvers

import (
	"algorithmia/src/models"
	"math"
)

// trajectoryCapLen je maksimalan broj zapisa po putanji posle downsample-a (pravilo 2)
const trajectoryCapLen = 200

// recordTrajectory dodaje TrajectoryPoint u niz SAMO ako se vrednost promenila u odnosu na poslednji već
// zabeleženi zapis (pravilo 1) — pošto niz uvek počinje prazan, prvi poziv uvek upisuje zapis bez obzira na
// forceRecord. forceRecord=true se koristi za obavezan završni zapis (posle glavne petlje, tik pre konstrukcije
// result-a) da bi se garantovao invarijant da poslednji zapis putanje odgovara finalnom Point/Value, čak i kad
// se vrednost nije promenila od poslednjeg zabeleženog koraka
func recordTrajectory(traj []models.TrajectoryPoint, step int, point []float64, value float64, forceRecord bool) []models.TrajectoryPoint {

	if !forceRecord && len(traj) > 0 && traj[len(traj)-1].Value == value {
		return traj
	}

	return append(traj, models.TrajectoryPoint{
		Step:  step,
		Point: append([]float64(nil), point...),
		Value: value,
	})
}

// capTrajectory downsample-uje niz na najviše maxLen zapisa ravnomernim biranjem svakog N-tog indeksa, ali UVEK
// zadržava POSLEDNJI zapis originalnog niza (pravilo 2) — garantuje da downsample nikad ne pokvari invarijant da
// poslednji zapis putanje odgovara finalnom rezultatu
func capTrajectory(traj []models.TrajectoryPoint, maxLen int) []models.TrajectoryPoint {

	if len(traj) <= maxLen {
		return traj
	}

	// mora biti ceiling deljenje (ne floor) — floor bi za len(traj) u opsegu (maxLen, 2*maxLen) davao n=1
	// (npr. 364/200=1 celobrojnom deljenju), što znači korak od 1 i NIKAKAV stvarni downsample, kršeći
	// garanciju od najviše maxLen zapisa
	n := (len(traj) + maxLen - 1) / maxLen
	if n < 1 {
		n = 1
	}

	capped := make([]models.TrajectoryPoint, 0, maxLen+1)
	for i := 0; i < len(traj); i += n {
		capped = append(capped, traj[i])
	}

	last := traj[len(traj)-1]
	if capped[len(capped)-1].Step != last.Step {
		capped = append(capped, last)
	}

	return capped
}

// proximalGradEps je podrazumevani korak za numGrad centralnu razliku, usklađen sa h=1e-5 koji već
// koristi functions.NumericalGradient
const proximalGradEps = 1e-5

// numGrad aproksimira gradijent funkcije f u tački x koristeći centralne razlike sa korakom eps po svakoj dimenziji
func numGrad(f func([]float64) float64, x []float64, eps float64) []float64 {

	gradient := make([]float64, len(x))
	for d := range x {
		plus := append([]float64(nil), x...)
		minus := append([]float64(nil), x...)
		plus[d] += eps
		minus[d] -= eps
		gradient[d] = (f(plus) - f(minus)) / (2 * eps)
	}

	return gradient
}

// proxL1 primenjuje soft-thresholding (proksimalni operator L1 norme) na svaku dimenziju vektora v
func proxL1(v []float64, threshold float64) []float64 {

	result := make([]float64, len(v))
	for d := range v {
		result[d] = proxL1Scalar(v[d], threshold)
	}

	return result
}

// proxL1Scalar primenjuje soft-thresholding na jednu skalarnu vrednost
func proxL1Scalar(v, threshold float64) float64 {

	magnitude := math.Abs(v) - threshold
	if magnitude < 0 {
		return 0
	}

	return math.Copysign(magnitude, v)
}

// l1Norm računa L1 normu (sumu apsolutnih vrednosti) vektora
func l1Norm(v []float64) float64 {

	sum := 0.0
	for _, x := range v {
		sum += math.Abs(x)
	}

	return sum
}

// clipStep ograničava svaku komponentu koraka delta na [-bound, bound] pre primene na poziciju, sprečavajući
// numeričku eksploziju na loše uslovljenim funkcijama (npr. rosenbrock) gde numerički gradijent može biti veliki
func clipStep(delta []float64, bound float64) []float64 {

	clipped := make([]float64, len(delta))
	for d := range delta {
		clipped[d] = clamp(delta[d], -bound, bound)
	}

	return clipped
}

// hessDiagEps je korak za numHessDiag centralnu razliku drugog reda — veći od proximalGradEps jer formula
// deli sa eps^2, pa je osetljivija na akumulaciju greške zaokruživanja pri premalom koraku
const hessDiagEps = 1e-3

// numHessDiag aproksimira dijagonalu Hesijana funkcije f u tački x centralnom razlikom drugog reda po svakoj
// dimenziji nezavisno (bez unakrsnih članova)
func numHessDiag(f func([]float64) float64, x []float64, eps float64) []float64 {

	fx := f(x)
	hess := make([]float64, len(x))
	for d := range x {
		plus := append([]float64(nil), x...)
		minus := append([]float64(nil), x...)
		plus[d] += eps
		minus[d] -= eps
		hess[d] = (f(plus) - 2*fx + f(minus)) / (eps * eps)
	}

	return hess
}

// backtrackAlphaCandidates su standardni koraci probani redom u jednostavnoj sekvencijalnoj line search pretrazi
var backtrackAlphaCandidates = []float64{1.0, 0.5, 0.25, 0.1}

// backtrackAlpha vraća prvi alpha iz backtrackAlphaCandidates za koji f(x+alpha*p) < f(x); ako nijedan kandidat
// ne poboljšava vrednost, vraća poslednji (najmanji) kandidat kao siguran fallback korak
func backtrackAlpha(f func([]float64) float64, x, p []float64) float64 {

	fx := f(x)
	fallback := backtrackAlphaCandidates[len(backtrackAlphaCandidates)-1]
	for _, alpha := range backtrackAlphaCandidates {
		trial := make([]float64, len(x))
		for d := range trial {
			trial[d] = x[d] + alpha*p[d]
		}
		if f(trial) < fx {
			return alpha
		}
	}

	return fallback
}
