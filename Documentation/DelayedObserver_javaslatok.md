# DelayedObserver – átnézés és javaslatok

Dátum: 2026-11-13

Érintett fájlok (`DCmotor/`):
- `DCmotor_PINN_DelayedObserver.m`
- `DCmotor_modelLoss_DelayedObserver.m`
- adatforrás: `DCmotor_ODE45_reference_DynamicLoad.mat` (konstans U0, lépcsős terhelés)

Ez a jegyzet a kód átnézéséből született, a kódon nem változtattam. A nagyságrendi állítások becslések, nincsenek külön lefuttatva. A log-számok a korábbi futásból (10000 iteráció) származnak.

## Az implementáció röviden

- Feedforward háló, az idő **nem** bemenet.
- Bemenet (9): `U`, `i_meas`, `omega_meas` a `k`, `k-1`, `k-2` időpontokban.
- Kimenet (3): `i_hat(k)`, `omega_hat(k)`, `M_L_hat(k)`.
- Fizika: a deriváltat a háló kimenetén, szomszédos oszlopokból számolja (backward Euler), a maradék a `2:end` indexeken van felírva.
- A terhelés (`M_L`) a tanításban **nem** szerepel címkeként. Csak a mechanikai maradékból és a `loadChangeLoss` regularizációból tanulja a háló.
- Utolsó ismert eredmény (10000 iteráció, tanító adatsoron): RMSE áram 0.0046 A, szögsebesség 0.0069 rad/s, terhelés 0.0125 Nm (a lépcső 0.1 Nm).

## Fontos

### 1. A terhelés-regularizáció dominálhatja a veszteséget
- A korábbi logban `Load Change Loss ≈ 4`, `lambdaLoadChange = 2e-4`, így a tag ≈ 8e-4. Ehhez képest Physics ≈ 1.8e-4, Data ≈ 3e-5, Total ≈ 1.07e-3.
- A kódban most `lambdaLoadChange = 1e-3`, ezért a tag kb. 5×-ösre nő. Valószínűleg ez határozza meg a terhelés alakját, nem a fizika.
- A tag `sum` (nem `mean`) és |Δ|^0.5 jellegű, tehát értéke függ a minták számától (`N`). Más `dt`-nél vagy hosszabb adatsornál újra kell hangolni a lambdát.
- Javaslat:
  - `mean` használata a `sum` helyett.
  - A log kiírásakor `lambda * tag` szorzatokat is érdemes nézni.
  - `lambdaLoadChange` végigpróbálása (pl. 0, 1e-5, 1e-4, 1e-3).
- A lépcsők lekerekedése és a ~12%-os terhelés-RMSE valószínűleg itt dől el.

### 2. A mechanikai maradék súlya kicsi
- Becslés: `resOmega²` ~1e-2, `resI²` ~1 nagyságrendű. A terhelés csak `resOmega`-ban szerepel.
- Javaslat: maradékok skálázása (pl. `resOmega / 0.1`) vagy külön lambda az egyes tagokra. A projektterv is ezt írja elő (`s_e`, `s_m` skálák).

### 3. Nincs független teszt
- A kiértékelés a tanító adatsoron történik (`predict(net, xInput)`), tehát csak illeszkedést mutat, általánosítást nem.
- A korábbi (bonyolultabb) verzióban volt külön nem látott teszt adatsor, az egyszerűsítésnél kiesett.
- Javaslat: második `.mat` fájl más terhelésprofillal (más időpontok, más amplitúdó), és az RMSE kiírása azon is.

### 4. A feszültség-bemenet most felesleges
- `U` konstans, így a 3 feszültség-oszlop konstans, nem hordoz információt.
- Amíg konstans, elhagyható a bemenetből (6 bemenet marad).
- Ha később változó lesz, a bemenet jó, de a veszteségben is vektorként kell átadni (lásd 5. pont).

### 5. `U` a veszteségben skalár
- `resI`-ben `U` skalár, a maradékok viszont a `2:end` indexeken értelmezettek.
- Változó feszültségnél `U` vektor kell, a `tValid` szerint igazítva (a `k`-adik maradékhoz a `k`-adik mintához tartozó `U`). Különben egy indexeltolás csendben rossz eredményt ad.

## Közepesen fontos

### 6. Identitás-hajlam, zajkezelés
- A bemeneten ott van `i(k)`, `omega(k)`, a kimenet és az adatcél ugyanez. Zajmentesen az adatveszteség triviálisan kicsi, nem bizonyít semmit.
- Zajnál a háló a zajt is átmásolhatja.
- Javaslat: zajos teszthez a bemenetre kerüljön zaj, a cél maradjon tiszta (vagy a hatást külön vizsgáljuk). Zajnál több késleltetés (`d > 2`) is kell.

### 7. Hiányzó baseline
- Zajmentesen a terhelés közvetlenül kiszámolható: `M_L = Kt*i - b*omega - J*d(omega)/dt` (véges differencia a mért adatokon).
- Ezzel kell összevetni a PINN-t, különben nem látszik, hogy a háló ad-e hozzá bármit. Zajos adatnál lesz érdekes az összevetés.
- A projektterv (validációs fejezet) is kér összehasonlítást.

### 8. Elsőrendű (backward Euler) derivált
- `dt ≈ 0.01 s`-nél a hiba várhatóan kicsi, de szisztematikus eltolást adhat a terhelésben a lépcsők körül.
- Ha a hiba nem megy lejjebb: trapéz-szabály (`(x(k)+x(k-1))/2` a maradékban), ez egy sornyi változtatás.

### 9. Tanulási ráta, zajos végállapot
- A log szerint a veszteség nem monoton (pl. 9400-nál 8.2e-4, a 10000. iterációnál 1.07e-3), a konstans `1e-3` ráta miatt.
- Az RMSE az utolsó iteráció állapotából jön, ezért ingadozhat.
- Javaslat: lépcsős rátacsökkentés (mint a rugó–tömeg szkriptben), vagy a legjobb loss-hoz tartozó háló mentése.

### 10. Az első minta kimarad a fizikából
- A maradék csak a `2:end` indexekre van felírva, az első kimenet (főleg `M_L(1)`) csak a regularizációból kap vezérlést.
- Kis hatás, a széleken látszhat.

## Apróságok

- A minták sorrendje és a teljes batch kötött (a deriváltak szomszédos oszlopokból jönnek), ezért minibatch/keverés nem használható. Érdemes megjegyzésben jelezni.
- A kezdeti feltétel és a pozíció elmarad (`i`, `omega` a mérésből jön). Ez a modell `theta`-t nem becsüli, ezt a dokumentációban jelezni kell.
- A szkript a `DCmotor_ODE45_reference_DynamicLoad.mat`-ra támaszkodik, és a `DCmotor/` mappából kell indítani (vagy `-sd DCmotor`).
- A gyökérben van egy duplikált `DCmotor_ODE45_reference_ChangingVoltage.mat` (a `DCmotor/` mappában is megvan).

## Javasolt sorrend

1. **1., 2., 9. pont:** veszteségtagok skálázása és kiírása, `mean`, ráta-csökkentés.
2. **3. pont:** nem látott teszt adatsor.
3. **7. pont:** analitikus baseline.
4. **6., 4., 5. pont:** zaj, majd változó feszültség (az `U` vektorként kezelésével).

## Státusz

| # | Pont | Állapot |
|---|------|---------|
| 1 | Regularizáció dominanciája | nyitott |
| 2 | Maradékok skálázása | nyitott |
| 3 | Független teszt adatsor | nyitott |
| 4 | Felesleges feszültség-bemenet | nyitott |
| 5 | `U` vektorként a veszteségben | nyitott |
| 6 | Zaj / identitás-hajlam | nyitott |
| 7 | Analitikus baseline | nyitott |
| 8 | Trapéz-szabály | nyitott |
| 9 | Ráta-csökkentés / legjobb háló mentése | nyitott |
| 10 | Első minta a fizikában | nyitott |
