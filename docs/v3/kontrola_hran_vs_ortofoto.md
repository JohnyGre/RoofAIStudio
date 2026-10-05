# Kontrola hrán proti ortofotu (vizuálne overenie)

Metóda: model sa prekreslí na ortofoto (`output/v3/*_kontrola_hran_na_ortofote.png`),
obrázok sa dá vizuálne posúdiť (AutoGLM vision) a podľa nálezu sa opraví engine.

## Kolo 1 — stav pred opravou

Nález:
1. Žlté (odkvapy) posunuté mimo obrysu strechy, časť vedie **po streche** (zbytočné línie vnútri).
2. Modrá (hrebeň) posunutá a v skutočnosti kopíruje **úžľabie** → chybná klasifikácia.
3. Zelené (nárožia) sedia len čiastočne.
4. Červené (úžľabia) **úplne chýbajú**, hoci na streche sú viditeľné aspoň dve.

Opravy vykonané v engine:
- konvexnosť sa počíta z **maxima výšky** (maximum na hrane = hrebeň, minimum = úžľabie),
- odkvapy sa berú z **obrysu celej strechy** (union polygónov), nie z obrysu každej roviny zvlášť,
- hrany hrebeň/nárožie/úžľabie sa berú **presne z priesečníc rovín** (nie z obrysu bodov).

## Kolo 2 — stav po oprave

Nález:
1. Žlté: časť sedí (ľavý horný okraj), pravá a ľavá dolná časť **posunuté** dovnutra/mimo.
2. Červené: nakreslené, ale **nesprávne** — tvoria umelé obdĺžniky, nezodpovedajú skutočným úžľabiam.
3. Modré (hrebeň) a zelené (nárožia): **úplne chýbajú**.
4. Skutočné hrebene aj vonkajšie nárožia na snímke zostávajú neoznačené.

## Diagnóza (prečo to ešte nesedí)

- Model má 11 rovín pre jednu strechu, väčšina s plochou 11–52 m² → roviny sú **fragmenty**,
  nie skutočné strešné plochy. Párové testy medzi fragmentami dávajú nestabilný výsledok
  (preto „všetko úžľabie" v kole 2 a predtým „všetko hrebeň" v kole 1).
- Príčina fragmentácie: do mračna sa stále dostávajú susedné budovy v rade, lebo
  **chýba vektorový obrys budovy** (OSM teraz nevracia geometriu, ZBGIS vrstva je neoverená).
  Log to hlási ako `FOOTPRINT_GEOMETRY_APPROXIMATED`.
- Posun odkvapov: body siahajú za hranu strechy (presah), obrys z bodov je preto väčší.

## Ďalší krok (v poradí dôležitosti)

1. **Získať obrys budovy** (ZBGIS WFS alebo OSM s geometriou) a orezať mračno ním —
   tým sa fragmenty zlejú do skutočných plôch a párové testy začnú fungovať.
2. Až potom ladenie klasifikácie hrán; teraz by to bolo ladenie šumu.
3. Alternatíva, ak obrys nie je dostupný: odvodiť obrys z ortofota (Vision) a použiť ho
   ako dočasnú planimetrickú autoritu s príznakom v kontrakte.
