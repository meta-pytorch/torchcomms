typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	char			pme_ctrl;	/* S0 or S1 */
	char			__pad;
	int			pme_val;	/* S0/S1 encoding */
} pme_sparc_entry_t;

typedef struct {
	char			*mask_name;	/* mask name */
	char			*mask_desc;	/* mask description */
} pme_sparc_mask_t;

#define EVENT_MASK_BITS		8
typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	char			pme_ctrl;	/* S0 or S1 */
	char			__pad;
	int			pme_val;	/* S0/S1 encoding */
	pme_sparc_mask_t	pme_masks[EVENT_MASK_BITS];
} pme_sparc_mask_entry_t;

#define PME_CTRL_S0		1
#define PME_CTRL_S1		2
