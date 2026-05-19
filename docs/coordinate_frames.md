# Systèmes de coordonnées et conversions Lighthouse

Ce document décrit les deux systèmes de coordonnées en jeu sur le projet
hardware, la transformation qui les relie, et les modifications faites
dans le code pour passer de l'un à l'autre.

---

## 1. Pourquoi deux frames ?

La position absolue du drone vient du **Lighthouse**. La calibration du
Lighthouse définit l'**origine** et l'**orientation des axes** du repère
monde. Deux calibrations différentes → deux repères différents.

| Repère | Défini par | Utilisé par |
|--------|-----------|-------------|
| **Frame mission** | *Notre* calibration du Lighthouse | Tout notre code : `config.py`, la FSM, la géométrie, la perception, le CSV historique |
| **Frame examinateur** | La calibration du Lighthouse faite par le prof le jour de la démo | Le firmware du drone (état rapporté + commandes acceptées), le CSV des portes fourni par le prof |

On a développé, testé et calibré **tout le code en frame mission**. Le
jour de la démo, le drone parle en **frame examinateur**. Il faut donc
traduire entre les deux.

---

## 2. La transformation frame examinateur ↔ frame mission

La calibration du prof, exprimée par rapport à notre frame mission :

- **Nouvelle origine** placée au point mission `(LH_ORIGIN_SHIFT_X, 0, 0)`,
  avec `LH_ORIGIN_SHIFT_X = 1.0` m.
- **Axes x et y tournés de 180°** autour de l'axe z (rotation autour de
  la verticale).
- **Axe z inchangé.**

Autrement dit : `x_mission +x` ↔ `x_examinateur -x`, idem pour y, z commun.

### 2.1 Formules

Soit un point de coordonnées `(x, y, z)` dans une frame. Dans l'autre :

| Grandeur | Conversion | Raison |
|----------|-----------|--------|
| **Position** (point) | `x' = 1 - x` &nbsp; `y' = -y` &nbsp; `z' = z` | translation d'origine + rotation 180° |
| **Vecteur libre** (vitesse, accélération, offset relatif) | `x' = -x` &nbsp; `y' = -y` &nbsp; `z' = z` | rotation seule — un vecteur n'est pas affecté par la translation d'origine |
| **Cap (yaw)** | `yaw' = yaw + π` (wrappé dans [-π, π]) | rotation des axes de 180° |
| **roll, pitch** | inchangés | angles du corps, insensibles à une redéfinition du yaw monde |

### 2.2 Propriété clé : la transformation est une involution

Appliquer la transformation deux fois redonne la valeur d'origine :

- Position : `1 - (1 - x) = x`
- Yaw : `yaw + π + π = yaw + 2π ≡ yaw`

Conséquence pratique : **la même fonction sert dans les deux sens**
(examinateur → mission *et* mission → examinateur). Pas besoin de deux
jeux de fonctions.

### 2.3 Implémentation

Définie dans `config.py` (bloc « Lighthouse frame compatibility ») :

```python
LH_ORIGIN_SHIFT_X = 1.0

def frame_conv_pos(x, y, z):   # point : translation + rotation 180°
    return (LH_ORIGIN_SHIFT_X - x, -y, z)

def frame_conv_vec(x, y, z):   # vecteur libre : rotation seule
    return (-x, -y, z)

def frame_conv_yaw(yaw):       # cap, wrappé dans [-pi, pi]
    return atan2(sin(yaw + pi), cos(yaw + pi))
```

Ces fonctions sont aussi sûres en **vectorisé numpy** (utilisées telles
quelles sur les colonnes du CSV).

---

## 3. Stratégie : convertir aux frontières, pas partout

Le code interne (FSM, géométrie, perception, `config.py`) **reste
entièrement en frame mission**. La conversion est confinée aux **points
d'entrée/sortie** avec le monde extérieur.

```
                      ┌─────────────────────────────────┐
   FIRMWARE DRONE      │      NOTRE CODE (frame mission)  │
   (frame examinateur) │                                 │
                       │   FSM · géométrie · perception  │
   état drone ───[conv]──▶  config.py · motion planner   │
   commandes  ◀──[conv]───                               │
                       │                                 │
   CSV portes ───[conv]──▶                               │
   (frame examinateur)  └─────────────────────────────────┘
```

**Trois frontières de conversion, une seule transformation :**

| Frontière | Sens | Où | Conversion |
|-----------|------|-----|-----------|
| État du drone | examinateur → mission | `crazyflie_link.py` `_on_pose_log` / `_on_vel_log` | pose + vitesse |
| Commandes de vol | mission → examinateur | `crazyflie_link.py` `go_to` / `send_full_state_setpoint` | position/vecteur + yaw |
| CSV des portes | examinateur → mission | `motion_planner.py` `construct_waypoints` | positions + theta |

### Pourquoi pas tout réécrire en frame examinateur ?

- Tous nos essais, logs et réglages sont en frame mission — les garder
  permet de comparer avec l'historique le jour de la démo.
- Réécrire `config.py` impose des pièges silencieux : par ex. `X_BOUNDS`
  et `Y_BOUNDS` sont des tuples `(min, max)`, et la conversion (fonction
  décroissante) **inverse l'ordre** — il faudrait réordonner à la main,
  sans quoi tous les tests de borne échouent silencieusement.
- L'approche frontière évite d'auditer chaque fichier : le code interne
  ne voit jamais la frame examinateur.

---

## 4. Subtilité : l'angle `theta` du CSV des portes

Le CSV fourni par le prof est dans la **frame examinateur**, mais avec une
convention d'angle particulière pour la colonne `theta` :

> `theta` est mesuré depuis l'axe **`-x`** (et non depuis `+x` comme
> habituellement).

Deux décalages de π s'appliquent donc à `theta` :

1. **Convention « depuis -x »** : pour obtenir le cap standard (depuis
   `+x`) en frame examinateur → `+π`.
2. **Conversion examinateur → mission** : rotation des axes de 180° → `+π`.

Total : `theta_mission = theta_csv + π + π = theta_csv + 2π ≡ theta_csv`.

### → Les deux π s'annulent

**`theta` du CSV est utilisé tel quel** : c'est déjà le cap standard
(depuis `+x`) en frame mission.

**Intuition :** l'axe `+x` de la frame mission *est* l'axe `-x` de la
frame examinateur (c'est exactement ce que fait la rotation de 180°).
Donc « mesurer depuis `-x` en examinateur » = « mesurer depuis `+x` en
mission ». Le prof a donné `theta` directement dans notre convention.

⚠️ Cela ne concerne **que la colonne `theta`**. Les positions `x, y, z`
du CSV sont des coordonnées normales en frame examinateur et passent par
`frame_conv_pos`. `width` et `height` sont des scalaires, inchangés.

| Colonne CSV | Conversion appliquée |
|-------------|---------------------|
| `x, y, z`   | `frame_conv_pos` (translation d'origine + rotation 180°) |
| `theta`     | aucune — utilisée telle quelle (juste wrappée dans [-π, π]) |
| `width, height` | aucune |

---

## 5. Modifications faites dans le code

### `config.py`
- Import `math` étendu : `atan2, cos, sin` ajoutés.
- Nouveau bloc « Lighthouse frame compatibility » : constante
  `LH_ORIGIN_SHIFT_X` et fonctions `frame_conv_pos`, `frame_conv_vec`,
  `frame_conv_yaw`. C'est le **foyer unique** de la logique de frame.

### `hardware/crazyflie_link.py`
- Alias `_conv_pos / _conv_vec / _conv_yaw` vers `config.frame_conv_*`.
- `_on_pose_log` : pose entrante convertie examinateur → mission
  (`roll`/`pitch` laissés intacts).
- `_on_vel_log` : vitesse convertie (rotation seule).
- `go_to` : cible convertie mission → examinateur. Cas absolu →
  `frame_conv_pos` + `frame_conv_yaw` ; cas relatif → `frame_conv_vec`
  seul (un delta de yaw est invariant par changement de frame).
- `send_full_state_setpoint` : `pos` via `frame_conv_pos`, `vel`/`acc`
  via `frame_conv_vec`, `yaw` via `frame_conv_yaw`.
- `takeoff` / `land` : aucune modification — ne manipulent qu'une
  altitude `z`, invariante par la transformation.

### `racing/motion_planner.py`
- `construct_waypoints` : après lecture du CSV, conversion examinateur →
  mission. Positions `x, y, z` via `config.frame_conv_pos` ; `theta`
  utilisé tel quel (voir §4) ; `width`/`height` inchangés.

### Fichiers volontairement non modifiés
`planning/fsm.py`, `planning/geometry.py`, `perception/gate_localizer.py`,
`main.py`, `racing/main.py`, `racing/race_fsm.py` — ils raisonnent
entièrement en frame mission et passent toujours par `crazyflie_link.py`
ou `config`, donc la conversion aux frontières les couvre automatiquement.

---

## 6. Si la calibration du prof change

La transformation est entièrement paramétrée par **une seule constante** :

- Décalage d'origine différent de 1.0 m → ajuster `LH_ORIGIN_SHIFT_X`
  dans `config.py`.
- Si la rotation des axes n'est plus de 180° (cas non prévu ici), il
  faudrait généraliser `frame_conv_pos` / `frame_conv_vec` /
  `frame_conv_yaw` avec une matrice de rotation paramétrable.

### Vérification avant la démo
Poser le drone au sol et vérifier dans `logs/.../trace.csv` que `x, y,
yaw` correspondent bien aux valeurs attendues en frame mission.
